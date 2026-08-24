"""Render paper section 4 from result JSON files."""

from __future__ import annotations

from src.config import ROOT, load_config
from src.io_utils import read_json


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def acc_ci(node: dict) -> str:
    lo, hi = node["ci95"]
    return f"{node['acc']:.3f} (95% CI {lo:.3f}–{hi:.3f})"


def render() -> str:
    cfg = load_config()
    summary = read_json(ROOT / cfg["paths"]["eval_summary"])
    lines = ["### 4.1 Behavioral eval", ""]
    lines.append(f"Model: `{summary['model_id']}`. n = {summary['n_rows']} generations.")
    lines.append("")
    lines.append("| Condition | Overall acc | Δ vs neutral | encoding | code | math |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    neu = summary["by_condition"]["neutral"]["overall"]["acc"]
    for cond, block in summary["by_condition"].items():
        delta = "" if cond == "neutral" else f"{neu - block['overall']['acc']:+.3f}"
        lines.append(
            f"| {cond} | {acc_ci(block['overall'])} | {delta} | "
            f"{block['encoding']['acc']:.3f} | {block['code']['acc']:.3f} | {block['math']['acc']:.3f} |"
        )
    lines.append("")
    deltas = summary.get("delta_vs_neutral", {})
    if deltas:
        worst = max(deltas, key=lambda c: deltas[c]["overall"])
        lines.append(
            f"The largest overall gap versus neutral is **{worst}** "
            f"(Δ = {deltas[worst]['overall']:.3f})."
        )
    lines.append("")

    elicit_path = ROOT / cfg["paths"]["elicit"]
    if elicit_path.exists():
        elicit = read_json(elicit_path)
        lines += ["### 4.2 Elicitation", ""]
        lines.append("| Intervention | Acc on eval-framed test items |")
        lines.append("|---|---:|")
        for name, block in elicit["by_elicit"].items():
            acc = block["by_condition"]["eval"]["overall"]["acc"]
            lines.append(f"| {name} | {acc:.3f} |")
        lines.append("")

    suf_path = ROOT / cfg["paths"]["suffixes"]
    if suf_path.exists():
        suf = read_json(suf_path)
        lines += ["### 4.3 Adversarial suffixes", ""]
        lines.append(
            f"Searched {suf['n_candidates']} suffixes. "
            f"Train base acc {suf['train_base_acc']:.3f}; "
            f"winner train drop {suf['winner']['train_drop']:.3f} "
            f"(well-formed {suf['winner']['well_formed']:.2f}). "
            f"Held-out drop {suf['test_drop']:.3f} "
            f"(base {suf['test_base_acc']:.3f} → {suf['test_acc']:.3f}, "
            f"well-formed {suf['test_well_formed']:.2f})."
        )
        lines.append("")
        lines.append(f"Winning suffix: `{suf['winner']['suffix']}`")
        lines.append("")

    interp_path = ROOT / cfg["paths"]["interp"]
    if interp_path.exists():
        interp = read_json(interp_path)
        lines += ["### 4.4 Probes, steering, and patching", ""]
        lines.append(
            f"Max probe AUC = {interp['max_probe_auc']:.3f} at hidden-state index "
            f"{interp['best_probe_layer']} (decoder layer {interp['decoder_steer_layer']}). "
            f"Mean AUC across layers = {interp['mean_probe_auc']:.3f}."
        )
        lines.append("")
        lines.append("| α | induce (add v on neutral) | abolish (subtract v on eval) | random on neutral |")
        lines.append("|---:|---:|---:|---:|")
        for row in interp["steering"]:
            lines.append(
                f"| {row['alpha']} | {row['induce_on_neutral']:.3f} | "
                f"{row['abolish_on_eval']:.3f} | {row['random_on_neutral']:.3f} |"
            )
        lines.append("")
        if interp.get("patching"):
            lines.append("| Patch layer | Neutral acc after eval last-token patch |")
            lines.append("|---:|---:|")
            for row in interp["patching"]:
                lines.append(f"| {row['layer']} | {row['acc_neutral_with_eval_patch']:.3f} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    text = render()
    out = ROOT / "paper" / "results_auto.md"
    out.write_text(text + "\n")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
