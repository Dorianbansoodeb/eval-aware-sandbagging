from __future__ import annotations

from src.config import ROOT, load_config
from src.framings import PERSONA_SYSTEM, SYSTEM, user_prompt
from src.io_utils import read_jsonl, write_json
from src.model_utils import apply_chat, generate_batch, load_model
from src.score import extract_answer, is_correct, looks_well_formed, summarize


def _few_shot_prefix(items: list[dict], n_shots: int) -> str:
    shots = [it for it in items if it["split"] == "train"][:n_shots]
    parts = ["Here are solved examples:"]
    for s in shots:
        parts.append(f"Task: {s['task']}\nFinal answer: {s['gold']}")
    return "\n\n".join(parts) + "\n\nNow solve this new task.\n\n"


def run_elicit(cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    items = read_jsonl(ROOT / cfg["paths"]["items"])
    bundle = load_model(cfg["model_id"], cfg["dtype"], cfg["device"])
    n_shots = int(cfg["elicit"]["n_shots"])
    batch_size = int(cfg["batch_size"])
    max_new = int(cfg["max_new_tokens"])
    target_items = [it for it in items if it["split"] == "test"]

    jobs = []
    for item in target_items:
        base_task = item["task"]
        jobs.append(
            {
                "item": item,
                "condition": "eval",
                "elicit": "none",
                "prompt": apply_chat(bundle.tokenizer, user_prompt(base_task, "eval"), SYSTEM),
                "prefill": "",
            }
        )
        fs_task = _few_shot_prefix(items, n_shots) + base_task
        jobs.append(
            {
                "item": item,
                "condition": "eval",
                "elicit": "few_shot",
                "prompt": apply_chat(bundle.tokenizer, user_prompt(fs_task, "eval"), SYSTEM),
                "prefill": "",
            }
        )
        jobs.append(
            {
                "item": item,
                "condition": "eval",
                "elicit": "prefill",
                "prompt": apply_chat(bundle.tokenizer, user_prompt(base_task, "eval"), SYSTEM),
                "prefill": "Final answer:",
            }
        )
        jobs.append(
            {
                "item": item,
                "condition": "eval",
                "elicit": "persona",
                "prompt": apply_chat(
                    bundle.tokenizer, user_prompt(base_task, "eval"), PERSONA_SYSTEM
                ),
                "prefill": "",
            }
        )

    rows = []
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        comps = generate_batch(
            bundle,
            [j["prompt"] for j in chunk],
            max_new,
            prefills=[j["prefill"] for j in chunk],
        )
        for job, comp in zip(chunk, comps):
            pred = extract_answer(comp)
            rows.append(
                {
                    "id": job["item"]["id"],
                    "domain": job["item"]["domain"],
                    "condition": job["condition"],
                    "elicit": job["elicit"],
                    "gold": job["item"]["gold"],
                    "pred": pred,
                    "correct": is_correct(pred, job["item"]["gold"]),
                    "well_formed": looks_well_formed(comp),
                    "completion": comp,
                }
            )
        print(f"elicit {min(i + batch_size, len(jobs))}/{len(jobs)}")

    by_elicit = {}
    for name in ["none", "few_shot", "prefill", "persona"]:
        subset = [r for r in rows if r["elicit"] == name]
        by_elicit[name] = summarize(subset)
    out = {"n": len(target_items), "by_elicit": by_elicit}
    write_json(ROOT / cfg["paths"]["elicit"], out)
    return out
