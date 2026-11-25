"""Command line: python -m riskaudit --data data/financial_risk_assessment.csv --out results"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import audit  # noqa: E402
from .data import load, split_xy  # noqa: E402
from .models import candidates  # noqa: E402


def plot_scores(scores, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    s = scores.sort_values("balanced_accuracy")
    colors = ["#9aa5b1" if "baseline" in m else "#2f6fb0" for m in s["model"]]
    ax.barh(s["model"], s["balanced_accuracy"], xerr=s["balanced_accuracy_std"], color=colors)
    ax.axvline(1 / 3, color="#c0392b", linestyle="--", label="Chance (1/3)")
    ax.set_xlabel("Balanced accuracy (5-fold cross-validation)")
    ax.set_xlim(0, 0.6)
    ax.legend(loc="lower right")
    ax.set_title("No model beats chance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation(perm, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(perm.null_scores, bins=25, color="#9aa5b1", label="Shuffled labels (null)")
    ax.axvline(perm.score, color="#2f6fb0", linewidth=2.5, label=f"Real labels: {perm.score:.3f}")
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel("Permutations")
    ax.set_title(f"Permutation test, p = {perm.p_value:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_effects(assoc, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    a = assoc.sort_values("effect_size")
    ax.barh(a["feature"], a["effect_size"], color="#2f6fb0")
    ax.axvline(0.1, color="#c0392b", linestyle="--", label="Small effect threshold (0.1)")
    ax.set_xlim(0, 0.12)
    ax.set_xlabel("Effect size against the risk rating (bias-corrected Cramer's V or epsilon-squared)")
    ax.legend(loc="lower right")
    ax.set_title("Every feature on its own")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/financial_risk_assessment.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load(args.data)
    X, y = split_xy(df)

    assoc = audit.association_tests(df)
    scores = audit.cross_validated_scores(X, y, seed=args.seed)
    models = candidates(args.seed)
    best_name = scores[~scores["model"].str.contains("baseline")].sort_values("balanced_accuracy").iloc[-1]["model"]
    perm = audit.permutation_test(models[best_name], X, y, n_permutations=args.permutations, seed=args.seed)
    control = {s: audit.positive_control(X, y, models["Logistic Regression"], strength=s, seed=args.seed)
               for s in (0.25, 0.5, 1.0)}

    result = {
        "rows": int(len(df)),
        "class_share": y.value_counts(normalize=True).round(4).to_dict(),
        "missing_share": df.isna().mean().round(4)[df.isna().mean() > 0].to_dict(),
        "models": scores.round(4).to_dict(orient="records"),
        "best_model": best_name,
        "permutation_test": {"balanced_accuracy": round(perm.score, 4), "null_mean": round(perm.null_mean, 4),
                             "null_95th_percentile": round(perm.null_95th, 4), "p_value": round(perm.p_value, 4),
                             "permutations": args.permutations},
        "positive_control_balanced_accuracy": {str(k): round(v, 4) for k, v in control.items()},
        "association_tests": assoc.round(5).to_dict(orient="records"),
        "verdict": audit.verdict(perm, assoc),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    plot_scores(scores, out / "model_scores.png")
    plot_permutation(perm, out / "permutation_test.png")
    plot_effects(assoc, out / "feature_effects.png")

    print(scores[["model", "balanced_accuracy", "macro_f1", "accuracy"]].round(3).to_string(index=False))
    print(f"\nPermutation test ({best_name}): real {perm.score:.3f}, shuffled mean {perm.null_mean:.3f}, "
          f"p = {perm.p_value:.3f}")
    print("Positive control (planted feature strength -> balanced accuracy):",
          {k: round(v, 3) for k, v in control.items()})
    print("\nVerdict:", result["verdict"]["summary"])
    return result


if __name__ == "__main__":
    main()
