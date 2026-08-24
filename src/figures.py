from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import ROOT, load_config
from src.io_utils import read_json


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def plot_eval(summary: dict, out: Path) -> None:
    _style()
    table = summary["by_condition"]
    conditions = list(table.keys())
    domains = [k for k in next(iter(table.values())).keys() if k != "overall"]
    x = range(len(conditions))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, dom in enumerate(domains):
        xs = [xi + (i - (len(domains) - 1) / 2) * width for xi in x]
        ys = [table[c][dom]["acc"] for c in conditions]
        ax.bar(xs, ys, width=width, label=dom)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Paired framing eval ({summary.get('model_id', '')})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_probe(interp: dict, out: Path) -> None:
    _style()
    aucs = interp["probe_auc_by_layer"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(range(len(aucs)), aucs, marker="o", linewidth=1.5, markersize=3)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Held-out AUC")
    ax.set_title("Linear probe: eval vs neutral last-token residual")
    ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_steer(interp: dict, out: Path) -> None:
    _style()
    rows = interp["steering"]
    alphas = [r["alpha"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(alphas, [r["induce_on_neutral"] for r in rows], marker="o", label="Add v on neutral")
    ax.plot(alphas, [r["abolish_on_eval"] for r in rows], marker="s", label="Subtract v on eval")
    ax.plot(alphas, [r["random_on_neutral"] for r in rows], marker="^", label="Random direction on neutral")
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Activation steering at decoder layer {interp['decoder_steer_layer']}")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_elicit(elicit: dict, out: Path) -> None:
    _style()
    names = list(elicit["by_elicit"].keys())
    accs = [elicit["by_elicit"][n]["by_condition"]["eval"]["overall"]["acc"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(names, accs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy on eval-framed test items")
    ax.set_title("Elicitation on evaluation-framed prompts")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(ROOT / cfg["paths"]["eval_summary"])
    plot_eval(summary, fig_dir / "eval_accuracy.png")
    elicit_path = ROOT / cfg["paths"]["elicit"]
    if elicit_path.exists():
        plot_elicit(read_json(elicit_path), fig_dir / "elicit.png")
    interp_path = ROOT / cfg["paths"]["interp"]
    if interp_path.exists():
        interp = read_json(interp_path)
        plot_probe(interp, fig_dir / "probe_auc.png")
        plot_steer(interp, fig_dir / "steering.png")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
