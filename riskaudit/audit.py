"""The audit itself: univariate association tests, cross-validated models, a permutation test and a positive control."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, permutation_test_score

from .data import CATEGORICAL, LOCATION, NUMERIC, TARGET
from .models import candidates

SCORING = {"balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro", "accuracy": "accuracy"}


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate across many tests)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def cramers_v(table: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013).

    The plain version is inflated for columns with many categories (City has ~10,000 levels and would look
    like the strongest feature purely because of its size); the correction removes that bias.
    """
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    if n <= 1 or min(r, k) < 2:
        return 0.0
    phi2 = max(chi2 / n - (k - 1) * (r - 1) / (n - 1), 0.0)
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def association_tests(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Test every feature against the label on its own.

    Categorical and low-cardinality columns: chi-squared test, effect size Cramer's V.
    Continuous columns: Kruskal-Wallis test, effect size epsilon-squared.
    """
    features = features or NUMERIC + CATEGORICAL + LOCATION
    rows = []
    for col in features:
        sub = df[[col, TARGET]].dropna()
        categorical = sub[col].dtype == object or str(sub[col].dtype) in ("category", "string", "str") \
            or sub[col].nunique() <= 12
        if categorical:
            table = pd.crosstab(sub[col], sub[TARGET])
            chi2, p, dof, _ = stats.chi2_contingency(table)
            rows.append({"feature": col, "test": "chi-squared", "statistic": chi2, "p_value": p,
                         "effect_size": cramers_v(table), "effect_measure": "Cramer's V"})
        else:
            groups = [g[col].to_numpy() for _, g in sub.groupby(TARGET)]
            h, p = stats.kruskal(*groups)
            n, k = len(sub), len(groups)
            eps2 = max((h - k + 1) / (n - k), 0.0)
            rows.append({"feature": col, "test": "Kruskal-Wallis", "statistic": h, "p_value": p,
                         "effect_size": eps2, "effect_measure": "epsilon-squared"})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out.sort_values("p_value").reset_index(drop=True)


def cross_validated_scores(X, y, models=None, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    models = models or candidates(seed)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for key in SCORING:
            row[key] = float(np.mean(res[f"test_{key}"]))
            row[f"{key}_std"] = float(np.std(res[f"test_{key}"]))
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class PermutationResult:
    score: float
    null_mean: float
    null_95th: float
    p_value: float
    null_scores: np.ndarray


def permutation_test(model, X, y, n_permutations: int = 100, folds: int = 5, seed: int = 0) -> PermutationResult:
    """Refit the model on shuffled labels many times. If the real score is not above that null distribution,
    the features carry no usable information about the label."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    score, null, p = permutation_test_score(clone(model), X, y, cv=cv, scoring="balanced_accuracy",
                                            n_permutations=n_permutations, random_state=seed, n_jobs=-1)
    return PermutationResult(float(score), float(null.mean()), float(np.quantile(null, 0.95)), float(p), null)


def positive_control(X, y, model, strength: float = 1.0, folds: int = 5, seed: int = 0) -> float:
    """Add one synthetic feature that is weakly related to the label and check the pipeline picks it up.

    This proves a low score comes from the data, not from a broken pipeline. Returns balanced accuracy.
    """
    rng = np.random.default_rng(seed)
    codes = pd.Categorical(y).codes.astype(float)
    X2 = X.copy()
    X2["planted_signal"] = strength * codes + rng.normal(0, 1, len(codes))
    model = clone(model)
    prep = model.named_steps["prep"]
    prep.transformers = [(name, tr, list(cols) + (["planted_signal"] if name == "num" else []))
                         for name, tr, cols in prep.transformers]
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    res = cross_validate(model, X2, y, cv=cv, scoring="balanced_accuracy")
    return float(np.mean(res["test_score"]))


def verdict(perm: PermutationResult, assoc: pd.DataFrame, alpha: float = 0.05) -> dict:
    significant = assoc.loc[assoc["p_holm"] < alpha, "feature"].tolist()
    has_signal = perm.p_value < alpha
    return {
        "signal_detected": bool(has_signal),
        "features_significant_after_correction": significant,
        "summary": ("The features predict the risk rating better than chance."
                    if has_signal else
                    "No model beats shuffled labels: the features carry no usable information about the risk rating."),
    }
