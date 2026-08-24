"""Export a compact JSON bundle for the static demo and Gradio app."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.config import ROOT, load_config
from src.framings import FRAMINGS
from src.io_utils import read_json, read_jsonl


def main() -> None:
    cfg = load_config()
    items = {it["id"]: it for it in read_jsonl(ROOT / cfg["paths"]["items"])}
    gens = read_jsonl(ROOT / cfg["paths"]["generations"])
    by_id: dict[str, dict] = defaultdict(dict)
    for row in gens:
        by_id[row["id"]][row["condition"]] = {
            "pred": row["pred"],
            "correct": bool(row["correct"]),
            "well_formed": bool(row["well_formed"]),
            "completion": (row.get("completion") or "")[:800],
        }

    export_items = []
    for iid, item in items.items():
        export_items.append(
            {
                "id": iid,
                "domain": item["domain"],
                "task": item["task"],
                "gold": item["gold"],
                "kind": item.get("kind", ""),
                "runs": by_id.get(iid, {}),
            }
        )
    export_items.sort(key=lambda x: (x["domain"], x["id"]))

    elicit = read_json(ROOT / cfg["paths"]["elicit"])
    elicit_flat = {
        name: block["by_condition"]["eval"]["overall"]["acc"]
        for name, block in elicit["by_elicit"].items()
    }

    bundle = {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "repo": "https://github.com/Dorianbansoodeb/eval-aware-sandbagging",
        "question": (
            "Do instruction-tuned open-weight models systematically underperform "
            "on skills they demonstrably possess when a prompt is framed as an "
            "evaluation, and is that gap causally mediated by a single residual-stream direction?"
        ),
        "finding": (
            "At 1.5B, no sandbagging: eval-like framings do not reduce accuracy. "
            "The grader treats 150 km as 150. A linear probe reads the eval/neutral "
            "contrast perfectly from layer 1, but steering and patching do not change "
            "task performance."
        ),
        "framings": FRAMINGS,
        "eval": read_json(ROOT / cfg["paths"]["eval_summary"]),
        "interp": read_json(ROOT / cfg["paths"]["interp"]),
        "elicit": elicit_flat,
        "suffixes": read_json(ROOT / cfg["paths"]["suffixes"]),
        "featured": ["math-017", "code-038", "enc-038"],
        "items": export_items,
    }

    out_docs = ROOT / "docs" / "data.json"
    out_js = ROOT / "docs" / "data.js"
    out_demo = ROOT / "demo" / "data.json"
    payload = json.dumps(bundle, indent=None, separators=(",", ":"))
    out_docs.parent.mkdir(parents=True, exist_ok=True)
    out_docs.write_text(payload)
    out_js.write_text("window.DEMO_DATA = " + payload + ";\n")
    out_demo.parent.mkdir(parents=True, exist_ok=True)
    out_demo.write_text(payload)
    print(f"Wrote {out_docs}, {out_js}, {out_demo} ({len(export_items)} items)")


if __name__ == "__main__":
    main()
