from __future__ import annotations

import gc

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ROOT, load_config
from src.framings import SYSTEM, user_prompt
from src.io_utils import read_jsonl, write_json
from src.model_utils import apply_chat, decoder_layers, generate_batch, load_model
from src.score import extract_answer, is_correct


def _hidden_tuple(output):
    return output[0] if isinstance(output, tuple) else output


def collect_last_token_states(bundle, prompts: list[str], batch_size: int) -> np.ndarray:
    """Return array [n, n_layers+1, d] of last-token hidden states."""
    tok = bundle.tokenizer
    model = bundle.model
    chunks = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(bundle.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple n_layers+1
        # last token = -1 because of left padding
        stacked = torch.stack([h[:, -1, :].float().cpu() for h in hs], dim=1)
        chunks.append(stacked.numpy())
        del out, enc
    return np.concatenate(chunks, axis=0)


def difference_in_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    v = pos.mean(axis=0) - neg.mean(axis=0)
    n = np.linalg.norm(v) + 1e-8
    return v / n


def probe_aucs(X_pos: np.ndarray, X_neg: np.ndarray, train_idx, test_idx) -> list[float]:
    n_layers = X_pos.shape[1]
    aucs = []
    y_train = np.array([1] * len(train_idx) + [0] * len(train_idx))
    y_test = np.array([1] * len(test_idx) + [0] * len(test_idx))
    for layer in range(n_layers):
        X_train = np.concatenate([X_pos[train_idx, layer], X_neg[train_idx, layer]], axis=0)
        X_test = np.concatenate([X_pos[test_idx, layer], X_neg[test_idx, layer]], axis=0)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, solver="liblinear"),
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, proba)))
    return aucs


def _steer_hook(alpha: float, vec: torch.Tensor):
    def hook(module, args, output):
        hidden = _hidden_tuple(output)
        steered = hidden + alpha * vec.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return hook


def generate_with_layer_hook(bundle, prompts, layer_idx, hook, max_new_tokens, batch_size):
    handle = decoder_layers(bundle.model)[layer_idx].register_forward_hook(hook)
    try:
        outs = []
        for i in range(0, len(prompts), batch_size):
            outs.extend(generate_batch(bundle, prompts[i : i + batch_size], max_new_tokens))
        return outs
    finally:
        handle.remove()


def _accuracy(completions, items) -> float:
    bits = [int(is_correct(extract_answer(c), it["gold"])) for c, it in zip(completions, items)]
    return float(sum(bits) / max(len(bits), 1))


def run_interp(cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    items = read_jsonl(ROOT / cfg["paths"]["items"])
    bundle = load_model(cfg["model_id"], cfg["dtype"], cfg["device"])
    bs = int(cfg["batch_size"])
    max_new = int(cfg["max_new_tokens"])

    def prompts_for(cond: str) -> list[str]:
        return [apply_chat(bundle.tokenizer, user_prompt(it["task"], cond), SYSTEM) for it in items]

    print("collecting hidden states")
    H_eval = collect_last_token_states(bundle, prompts_for("eval"), bs)
    H_neu = collect_last_token_states(bundle, prompts_for("neutral"), bs)

    n = len(items)
    n_train = int(cfg["interp"]["probe_train_frac"] * n)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n))
    aucs = probe_aucs(H_eval, H_neu, train_idx, test_idx)
    best_layer = int(max(range(len(aucs)), key=lambda i: aucs[i]))

    vecs = {}
    for layer in range(H_eval.shape[1]):
        vecs[layer] = difference_in_means(H_eval[:, layer], H_neu[:, layer])
    v = vecs[best_layer]

    rng = np.random.default_rng(cfg["seed"])
    rand = rng.normal(size=v.shape).astype(np.float32)
    rand = rand / (np.linalg.norm(rand) + 1e-8)

    n_steer = min(int(cfg["interp"]["n_steer_items"]), max(len(test_idx), 8))
    steer_items = [items[i] for i in test_idx[:n_steer]]
    neu_prompts = [
        apply_chat(bundle.tokenizer, user_prompt(it["task"], "neutral"), SYSTEM) for it in steer_items
    ]
    eval_prompts = [
        apply_chat(bundle.tokenizer, user_prompt(it["task"], "eval"), SYSTEM) for it in steer_items
    ]

    # Map hidden-state index to decoder layer: hidden_states[0] is embeddings.
    decoder_layer = min(max(best_layer - 1, 0), bundle.n_layers - 1)
    vec_t = torch.tensor(v, dtype=torch.float32)
    rand_t = torch.tensor(rand, dtype=torch.float32)

    steer_rows = []
    for alpha in cfg["interp"]["alphas"]:
        print(f"steering alpha={alpha} layer={decoder_layer}")
        comps = generate_with_layer_hook(
            bundle, neu_prompts, decoder_layer, _steer_hook(float(alpha), vec_t), max_new, bs
        )
        acc_ind = _accuracy(comps, steer_items)
        comps_e = generate_with_layer_hook(
            bundle, eval_prompts, decoder_layer, _steer_hook(-float(alpha), vec_t), max_new, bs
        )
        acc_abo = _accuracy(comps_e, steer_items)
        comps_r = generate_with_layer_hook(
            bundle, neu_prompts, decoder_layer, _steer_hook(float(alpha), rand_t), max_new, bs
        )
        acc_rand = _accuracy(comps_r, steer_items)
        steer_rows.append(
            {
                "alpha": float(alpha),
                "induce_on_neutral": acc_ind,
                "abolish_on_eval": acc_abo,
                "random_on_neutral": acc_rand,
            }
        )

    # Activation patching: eval last-token residual into neutral runs.
    patch_layers = list(range(0, bundle.n_layers, int(cfg["interp"]["patch_layers_stride"])))
    if decoder_layer not in patch_layers:
        patch_layers.append(decoder_layer)
    patch_items = steer_items[: min(16, len(steer_items))]
    patch_rows = []
    for layer in patch_layers:
        print(f"patching layer {layer}")
        acc = _patch_accuracy(bundle, patch_items, layer, max_new)
        patch_rows.append({"layer": int(layer), "acc_neutral_with_eval_patch": acc})

    del H_eval, H_neu
    gc.collect()

    out = {
        "model_id": bundle.model_id,
        "n_items": n,
        "n_layers": bundle.n_layers,
        "probe_auc_by_layer": aucs,
        "best_probe_layer": best_layer,
        "decoder_steer_layer": decoder_layer,
        "steering": steer_rows,
        "patching": patch_rows,
        "mean_probe_auc": float(np.mean(aucs)),
        "max_probe_auc": float(np.max(aucs)),
    }
    write_json(ROOT / cfg["paths"]["interp"], out)
    return out


def _patch_accuracy(bundle, items, layer_idx, max_new_tokens) -> float:
    tok = bundle.tokenizer
    layer = decoder_layers(bundle.model)[layer_idx]
    correct = 0
    for it in items:
        donor = apply_chat(bundle.tokenizer, user_prompt(it["task"], "eval"), SYSTEM)
        recip = apply_chat(bundle.tokenizer, user_prompt(it["task"], "neutral"), SYSTEM)
        donor_enc = tok(donor, return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            captured = {}

            def cap(module, args, output):
                h = _hidden_tuple(output)
                captured["vec"] = h[:, -1, :].detach()

            h1 = layer.register_forward_hook(cap)
            try:
                bundle.model(**donor_enc, use_cache=False)
            finally:
                h1.remove()
        vec = captured["vec"]
        prompt_len = int(tok(recip, return_tensors="pt")["input_ids"].shape[1])

        def patch(module, args, output):
            h = _hidden_tuple(output)
            if h.shape[1] >= 2 or h.shape[1] == prompt_len:
                h = h.clone()
                h[:, -1, :] = vec.to(device=h.device, dtype=h.dtype)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        h2 = layer.register_forward_hook(patch)
        try:
            comp = generate_batch(bundle, [recip], max_new_tokens)[0]
        finally:
            h2.remove()
        correct += int(is_correct(extract_answer(comp), it["gold"]))
    return float(correct / max(len(items), 1))
