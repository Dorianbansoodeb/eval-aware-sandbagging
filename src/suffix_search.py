from __future__ import annotations

import random

from src.config import ROOT, load_config
from src.framings import BANNED_SUFFIX_TERMS, SYSTEM, user_prompt
from src.io_utils import read_jsonl, write_json
from src.model_utils import apply_chat, generate_batch, load_model
from src.score import extract_answer, is_correct, looks_well_formed

VOCAB = [
    "please",
    "note",
    "context",
    "quietly",
    "consider",
    "ordinary",
    "routine",
    "aside",
    "meanwhile",
    "anyway",
    "perhaps",
    "simply",
    "continue",
    "detail",
    "remember",
    "focus",
    "steady",
    "neutral",
    "plain",
    "remark",
    "incidentally",
    "also",
    "still",
    "today",
    "here",
    "now",
    "kindly",
    "briefly",
    "clearly",
    "exactly",
    "toward",
    "item",
    "request",
    "user",
    "assistant",
    "follow",
    "next",
    "prior",
    "after",
    "before",
]


def _legal(suffix: str) -> bool:
    low = suffix.lower()
    return not any(term in low for term in BANNED_SUFFIX_TERMS)


def _acc(bundle, items, suffix, cfg) -> tuple[float, float]:
    prompts = [
        apply_chat(bundle.tokenizer, user_prompt(it["task"], "neutral", suffix=suffix), SYSTEM)
        for it in items
    ]
    comps = []
    bs = int(cfg["batch_size"])
    for i in range(0, len(prompts), bs):
        comps.extend(generate_batch(bundle, prompts[i : i + bs], int(cfg["max_new_tokens"])))
    bits = [int(is_correct(extract_answer(c), it["gold"])) for c, it in zip(comps, items)]
    formed = [int(looks_well_formed(c)) for c in comps]
    return sum(bits) / max(len(bits), 1), sum(formed) / max(len(formed), 1)


def run_suffix_search(cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    rng = random.Random(cfg["seed"] + 7)
    items = read_jsonl(ROOT / cfg["paths"]["items"])
    gens = read_jsonl(ROOT / cfg["paths"]["generations"])
    # Prefer items the model solved under neutral framing.
    solved = {r["id"] for r in gens if r["condition"] == "neutral" and r["correct"]}
    pool = [it for it in items if it["id"] in solved] or items
    n_train = int(cfg["suffix"]["n_train_items"])
    n_test = int(cfg["suffix"]["n_test_items"])
    train = pool[:n_train]
    test = pool[n_train : n_train + n_test] or pool[:n_test]

    bundle = load_model(cfg["model_id"], cfg["dtype"], cfg["device"])
    base_acc, base_formed = _acc(bundle, train, "", cfg)

    candidates = []
    n_cand = int(cfg["suffix"]["n_candidates"])
    slen = int(cfg["suffix"]["suffix_len"])
    while len(candidates) < n_cand:
        words = rng.sample(VOCAB, slen)
        suf = " ".join(words)
        if _legal(suf):
            candidates.append(suf)

    scored = []
    for i, suf in enumerate(candidates):
        acc, formed = _acc(bundle, train, suf, cfg)
        scored.append(
            {
                "suffix": suf,
                "train_acc": acc,
                "train_drop": base_acc - acc,
                "well_formed": formed,
            }
        )
        print(f"suffix {i + 1}/{len(candidates)} drop={base_acc - acc:.3f} formed={formed:.2f}")

    scored.sort(key=lambda x: (x["train_drop"] * (1 if x["well_formed"] >= 0.5 else 0)), reverse=True)
    winner = scored[0]
    test_base, _ = _acc(bundle, test, "", cfg)
    test_acc, test_formed = _acc(bundle, test, winner["suffix"], cfg)
    out = {
        "train_base_acc": base_acc,
        "train_base_formed": base_formed,
        "winner": winner,
        "test_base_acc": test_base,
        "test_acc": test_acc,
        "test_drop": test_base - test_acc,
        "test_well_formed": test_formed,
        "top5": scored[:5],
        "n_candidates": len(scored),
    }
    write_json(ROOT / cfg["paths"]["suffixes"], out)
    return out
