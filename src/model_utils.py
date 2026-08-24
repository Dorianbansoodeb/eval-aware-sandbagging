from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model_id: str
    model: Any
    tokenizer: Any
    device: torch.device
    n_layers: int


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(model_id: str, dtype_name: str = "float16", device_name: str = "auto") -> ModelBundle:
    os.environ.setdefault("HF_HOME", str(_hf_home()))
    device = pick_device(device_name)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    if device.type == "cpu":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    n_layers = int(getattr(model.config, "num_hidden_layers", len(model.model.layers)))
    return ModelBundle(model_id=model_id, model=model, tokenizer=tokenizer, device=device, n_layers=n_layers)


def _hf_home():
    from src.config import ROOT

    p = ROOT / ".cache" / "huggingface"
    p.mkdir(parents=True, exist_ok=True)
    return p


def apply_chat(tokenizer, user: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prefix = f"{system}\n\n" if system else ""
        return prefix + user + "\n\nAssistant:"


@torch.no_grad()
def generate_batch(
    bundle: ModelBundle,
    prompts: list[str],
    max_new_tokens: int,
    prefills: list[str] | None = None,
) -> list[str]:
    tok = bundle.tokenizer
    model = bundle.model
    texts = []
    for i, p in enumerate(prompts):
        pre = (prefills[i] if prefills else "") or ""
        texts.append(p + pre)

    encoded = tok(texts, return_tensors="pt", padding=True).to(bundle.device)
    out = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    completions = []
    for i, seq in enumerate(out):
        prompt_len = int(encoded["attention_mask"][i].sum().item())
        # left padding: generated tokens start after full padded length
        gen_ids = seq[encoded["input_ids"].shape[1] :]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        if prefills and prefills[i]:
            text = prefills[i] + text
        completions.append(text)
        _ = prompt_len  # kept for clarity / future patching
    return completions


def decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not find decoder layers on this model")
