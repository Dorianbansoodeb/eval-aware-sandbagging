from __future__ import annotations

import math
import re
from typing import Any

FINAL_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE)
BOX_RE = re.compile(r"\\boxed\{([^}]+)\}")
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")
# Trailing unit tokens after a number: "150 km", "90 kilometers", "23 mugs"
TRAILING_UNIT_RE = re.compile(
    r"^\s*[-+]?(?:\d+\.\d+|\d+)\s*[a-zA-Z%°/]*\s*$",
    re.IGNORECASE,
)


def extract_answer(text: str) -> str:
    if not text:
        return ""
    m = FINAL_RE.search(text)
    if m:
        return _clean(m.group(1).splitlines()[0])
    m = BOX_RE.search(text)
    if m:
        return _clean(m.group(1))
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return _clean(lines[-1]) if lines else ""


def _clean(s: str) -> str:
    s = s.strip().strip("`'\"")
    s = s.replace("```", "").strip()
    s = s.replace("$", "")
    if s.endswith("."):
        s = s[:-1]
    return " ".join(s.split())


def _as_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _numeric_value(s: str) -> float | None:
    """Parse a gold or pred that is a number, optionally with a unit."""
    s = _clean(s)
    direct = _as_float(s)
    if direct is not None:
        return direct
    if TRAILING_UNIT_RE.match(s):
        m = NUM_RE.search(s)
        if m:
            return _as_float(m.group(0))
    return None


def is_correct(pred: str, gold: str) -> bool:
    p = _clean(pred).lower()
    g = _clean(gold).lower()
    if not p or not g:
        return False
    if p == g:
        return True
    if p.replace(" ", "") == g.replace(" ", ""):
        return True
    g_num = _numeric_value(gold)
    p_num = _numeric_value(pred)
    if g_num is not None and p_num is not None:
        return math.isclose(p_num, g_num, rel_tol=0, abs_tol=1e-6)
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
        skill_bits = []
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
            if dom != "encoding":
                skill_bits.extend(bits)
        acc, lo, hi = bootstrap_ci(all_bits)
        table[cond]["overall"] = {"n": len(all_bits), "acc": acc, "ci95": [lo, hi]}
        if skill_bits:
            sacc, slo, shi = bootstrap_ci(skill_bits)
            table[cond]["math_code"] = {"n": len(skill_bits), "acc": sacc, "ci95": [slo, shi]}

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
            if "math_code" in table[cond] and "math_code" in table["neutral"]:
                deltas[cond]["math_code"] = (
                    table["neutral"]["math_code"]["acc"] - table[cond]["math_code"]["acc"]
                )
    return {"by_condition": table, "delta_vs_neutral": deltas, "n_rows": len(rows)}
