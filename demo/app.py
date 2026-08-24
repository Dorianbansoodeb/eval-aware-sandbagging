"""Local Gradio demo for resume walkthroughs. Uses committed generations by default."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((Path(__file__).resolve().parent / "data.json").read_text())
ITEMS = {it["id"]: it for it in DATA["items"]}
DOMAINS = ["math", "code", "encoding"]
CONDS = ["neutral", "eval", "threat", "anti_eval", "oversight"]
LABELS = {
    "neutral": "Neutral",
    "eval": "Eval benchmark",
    "threat": "Deployment threat",
    "anti_eval": "Not a test",
    "oversight": "Safety oversight",
}


def item_choices(domain: str) -> list[str]:
    return [f"{it['id']}  ·  {it['task'].splitlines()[0][:72]}" for it in DATA["items"] if it["domain"] == domain]


def parse_id(choice: str) -> str:
    return choice.split(" ", 1)[0]


def show(domain: str, choice: str):
    iid = parse_id(choice) if choice else DATA["featured"][0]
    it = ITEMS[iid]
    task = f"**Gold:** `{it['gold']}`\n\n{it['task']}"
    cards = []
    for cond in CONDS:
        run = it["runs"].get(cond, {})
        ok = "correct" if run.get("correct") else "wrong"
        formed = "tagged Final answer" if run.get("well_formed") else "no Final answer tag"
        cards.append(
            f"### {LABELS[cond]}\n"
            f"**{ok}** · pred `{run.get('pred', '')}` · {formed}\n\n"
            f"```\n{run.get('completion', '')}\n```"
        )
    return task, *cards


def update_items(domain: str):
    choices = item_choices(domain)
    return gr.update(choices=choices, value=choices[0])


def build() -> gr.Blocks:
    featured_math = item_choices("math")
    default = next((c for c in featured_math if c.startswith("math-017")), featured_math[0])

    with gr.Blocks(title="Eval-aware sandbagging demo") as demo:
        gr.Markdown(
            f"""
# Eval-aware sandbagging
**{DATA['model_id']}** · [repo]({DATA['repo']})

{DATA['question']}

**Finding.** {DATA['finding']}

This demo replays the committed generations (no GPU). Same item, five framings, model answers as recorded.
"""
        )
        with gr.Row():
            domain = gr.Dropdown(DOMAINS, value="math", label="Domain")
            item = gr.Dropdown(featured_math, value=default, label="Item")
        task = gr.Markdown()
        outs = [gr.Markdown() for _ in CONDS]
        domain.change(update_items, domain, item).then(show, [domain, item], [task, *outs])
        item.change(show, [domain, item], [task, *outs])
        demo.load(show, [domain, item], [task, *outs])
        gr.Markdown(
            """
## Headline numbers
| Slice | Neutral | Eval framing |
|---|---:|---:|
| Math+code accuracy | 0.575 | 0.625 |
| Encoding | 0.00 | 0.00 |
| Probe AUC from layer 1 | — | 1.00 |
| Steering / patching | no accuracy change | no accuracy change |
"""
        )
    return demo


if __name__ == "__main__":
    build().launch()
