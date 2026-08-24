from __future__ import annotations

import math
import re
from typing import Any

FINAL_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE)
BOX_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_answer(text: str) -> str:
    if not text:
        return ""
    m = FINAL_RE.search(text)
    if m:
        return _clean(m.group(1))
    m = BOX_RE.search(text)
    if m:
        return _clean(m.group(1))
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return _clean(lines[-1]) if lines else ""


def _clean(s: str) -> str:
    s = s.strip().strip("`'\"")
    s = s.replace("```", "").strip()
    if s.endswith("."):
        s = s[:-1]
    return " ".join(s.split())


def is_correct(pred: str, gold: str) -> bool:
    p = _clean(pred).lower()
    g = _clean(gold).lower()
    if p == g:
        return True
    if p.replace(" ", "") == g.replace(" ", ""):
        return True
    try:
        return math.isclose(float(p), float(g), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return False


def looks_well_formed(text: str) -> bool:
    return bool(FINAL_RE.search(text or ""))


def bootstrap_ci(bits: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    import numpy as np

    arr = np.asarray(bits, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    acc = float(arr.mean())
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return acc, lo, hi


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    formed: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["domain"])
        grouped[key].append(int(row["correct"]))
        formed[key].append(int(row.get("well_formed", looks_well_formed(row.get("completion", "")))))

    conditions = sorted({c for c, _ in grouped})
    domains = sorted({d for _, d in grouped})
    table = {}
    for cond in conditions:
        table[cond] = {}
        all_bits = []
        for dom in domains:
            bits = grouped[(cond, dom)]
            acc, lo, hi = bootstrap_ci(bits)
            table[cond][dom] = {
                "n": len(bits),
                "acc": acc,
                "ci95": [lo, hi],
                "well_formed": float(sum(formed[(cond, dom)]) / max(len(bits), 1)),
            }
            all_bits.extend(bits)
        acc, lo, hi = bootstrap_ci(all_bits)
        table[cond]["overall"] = {"n": len(all_bits), "acc": acc, "ci95": [lo, hi]}

    deltas = {}
    if "neutral" in table:
        for cond in conditions:
            if cond == "neutral":
                continue
            deltas[cond] = {
                "overall": table["neutral"]["overall"]["acc"] - table[cond]["overall"]["acc"]
            }
            for dom in domains:
                deltas[cond][dom] = table["neutral"][dom]["acc"] - table[cond][dom]["acc"]
    return {"by_condition": table, "delta_vs_neutral": deltas, "n_rows": len(rows)}
