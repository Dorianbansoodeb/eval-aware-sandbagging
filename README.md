# Eval-aware sandbagging

**Research question:** Do instruction-tuned open-weight models systematically underperform on skills they demonstrably possess when a prompt is framed as an evaluation, and is that gap causally mediated by a single residual-stream direction?

**Answer at 1.5B (this repo):** no behavioral sandbagging; a perfectly readable eval-vs-neutral residual direction that does **not** causally control accuracy.

**Live demo:** [dorianbansoodeb.github.io/eval-aware-sandbagging](https://dorianbansoodeb.github.io/eval-aware-sandbagging/) — pick an item, swap five framings, inspect the recorded answers. No GPU.

Local Gradio: `pip install -r demo/requirements.txt && PYTHONPATH=. python demo/app.py`

Primary model: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Paper: [`paper/paper.md`](paper/paper.md).

## Main results

| Test | Result |
|---|---|
| Paired framing eval (120 items × 5 framings) | Eval-like wrappers match or beat `neutral`. Math+code: 0.625 (`eval`) vs 0.575 (`neutral`). Encoding is 0/40 in every condition. |
| Elicitation | 4-shot helps math; prefill and persona do not. Encoding stays at floor. |
| Random suffixes (no eval vocabulary) | Accuracy drops track collapsed `Final answer:` formatting, not selective underperformance. |
| Linear probe | AUC 0.50 at embeddings, **1.00 from layer 1** (lexical prompt contrast). |
| Steering + last-token patching | No material accuracy change. Readable ≠ causal. |

Figures: [`results/figures/`](results/figures/).

## Reproduce

Python 3.11+ (tested on 3.13). Apple Silicon MPS, CUDA, or CPU. ~3GB for the model plus ~2GB for the venv.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
export HF_HOME="$PWD/.cache/huggingface"
python src/run.py --stage all
```

Or `bash scripts/run_all.sh`. Stages: `data`, `eval`, `elicit`, `suffix`, `interp`, `figures`.

Config lives in [`configs/default.yaml`](configs/default.yaml). Point `model_id` at a 7B instruct model to rerun the same tests at a more interesting scale.

## What is in the repo

```
docs/                # GitHub Pages demo (static, replays committed generations)
demo/app.py          # local Gradio explorer
src/build_items.py   # 120 machine-gradable items
src/eval_run.py      # five framings, greedy generate, exact-match grader
src/elicit.py        # few-shot, prefill, persona
src/suffix_search.py # random suffixes minus eval vocabulary
src/interp.py        # probes, difference-in-means, steer, patch
paper/paper.md       # preprint
results/             # committed tables, generations, figures
```

## Resume bullet

Designed a paired evaluation of evaluation-aware sandbagging on Qwen2.5-1.5B-Instruct (encoding, code, math; five framings); adversarially searched non-lexical suffixes and elicitation interventions; and tested linear mediation with residual-stream probes, activation steering, and patching. Found no sandbagging gap at this scale: eval-like framings did not reduce accuracy, a perfectly readable eval-vs-neutral direction (AUC 1.0 from layer 1) did not causally control performance, and suffix-induced drops were format collapse rather than selective underperformance. Interactive demo: https://dorianbansoodeb.github.io/eval-aware-sandbagging/

## License

MIT
