from __future__ import annotations

from src.config import ROOT, load_config
from src.framings import SYSTEM, user_prompt
from src.io_utils import read_jsonl, write_json, write_jsonl
from src.model_utils import apply_chat, generate_batch, load_model
from src.score import extract_answer, is_correct, looks_well_formed, summarize


def run_eval(cfg_path: str | None = None, model_id: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    items = read_jsonl(ROOT / cfg["paths"]["items"])
    mid = model_id or cfg["model_id"]
    bundle = load_model(mid, cfg["dtype"], cfg["device"])
    conditions = list(cfg["eval"]["conditions"])
    batch_size = int(cfg["batch_size"])
    max_new = int(cfg["max_new_tokens"])

    jobs = []
    for item in items:
        for cond in conditions:
            user = user_prompt(item["task"], cond)
            prompt = apply_chat(bundle.tokenizer, user, SYSTEM)
            jobs.append({"item": item, "condition": cond, "prompt": prompt})

    rows = []
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        completions = generate_batch(bundle, [j["prompt"] for j in chunk], max_new)
        for job, comp in zip(chunk, completions):
            pred = extract_answer(comp)
            rows.append(
                {
                    "id": job["item"]["id"],
                    "domain": job["item"]["domain"],
                    "condition": job["condition"],
                    "gold": job["item"]["gold"],
                    "pred": pred,
                    "correct": is_correct(pred, job["item"]["gold"]),
                    "well_formed": looks_well_formed(comp),
                    "completion": comp,
                    "model_id": mid,
                }
            )
        print(f"eval {min(i + batch_size, len(jobs))}/{len(jobs)}")

    out = ROOT / cfg["paths"]["generations"]
    if model_id:
        out = out.with_name(out.stem + f"_{mid.split('/')[-1]}" + out.suffix)
    write_jsonl(out, rows)
    summary = summarize(rows)
    summary["model_id"] = mid
    summary_path = ROOT / cfg["paths"]["eval_summary"]
    if model_id:
        summary_path = summary_path.with_name(
            summary_path.stem + f"_{mid.split('/')[-1]}" + summary_path.suffix
        )
    write_json(summary_path, summary)
    return summary
