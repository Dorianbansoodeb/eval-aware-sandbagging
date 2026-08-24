"""Re-grade committed generations with the current scorer (no model needed)."""

from __future__ import annotations

from src.config import ROOT, load_config
from src.io_utils import read_jsonl, write_json, write_jsonl
from src.score import extract_answer, is_correct, looks_well_formed, summarize


def rescore() -> dict:
    cfg = load_config()
    path = ROOT / cfg["paths"]["generations"]
    rows = read_jsonl(path)
    flipped = 0
    for row in rows:
        pred = extract_answer(row.get("completion", ""))
        correct = is_correct(pred, row["gold"])
        if bool(row.get("correct")) != correct:
            flipped += 1
        row["pred"] = pred
        row["correct"] = correct
        row["well_formed"] = looks_well_formed(row.get("completion", ""))
    write_jsonl(path, rows)
    summary = summarize(rows)
    summary["model_id"] = rows[0].get("model_id") if rows else None
    summary["rescored_flips"] = flipped
    write_json(ROOT / cfg["paths"]["eval_summary"], summary)
    print(f"rescored {len(rows)} rows, {flipped} labels flipped")
    return summary


if __name__ == "__main__":
    print(rescore())
