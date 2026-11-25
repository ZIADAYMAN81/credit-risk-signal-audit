import numpy as np
import pandas as pd
import pytest

from riskaudit import audit
from riskaudit.data import CATEGORICAL, LOCATION, NUMERIC, TARGET, SchemaError, split_xy, validate
from riskaudit.models import candidates


def synthetic(n=1200, signal=0.0, seed=0):
    """A dataset with the real schema. `signal` controls how strongly Credit Score depends on the label."""
    rng = np.random.default_rng(seed)
    y = rng.choice(["Low", "Medium", "High"], size=n, p=[0.6, 0.3, 0.1])
    code = pd.Series(y).map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    df["Credit Score"] = 700 - signal * 60 * code + rng.normal(0, 50, n)
    for c in CATEGORICAL:
        df[c] = rng.choice(["a", "b", "c"], size=n)
    for c in LOCATION:
        df[c] = rng.choice(["x", "y"], size=n)
    df.loc[rng.random(n) < 0.15, "Income"] = np.nan  # same missingness pattern as the real data
    df[TARGET] = y
    return df


def test_holm_matches_hand_calculation():
    adjusted = audit.holm([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.20])


def test_holm_is_monotone_and_capped():
    p = np.random.default_rng(1).random(30)
    adj = audit.holm(p)
    assert np.all(adj >= p) and np.all(adj <= 1)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_corrected_cramers_v_ignores_high_cardinality_noise():
    rng = np.random.default_rng(0)
    n = 3000
    labels = rng.choice(["a", "b", "c"], n)
    near_unique = pd.Series(rng.integers(0, 2000, n))  # like City: almost one row per category
    assert audit.cramers_v(pd.crosstab(near_unique, labels)) < 0.05
    perfect = pd.Series(labels)
    assert audit.cramers_v(pd.crosstab(perfect, labels)) == pytest.approx(1.0, abs=0.01)


def test_schema_validation():
    df = synthetic(50)
    with pytest.raises(SchemaError):
        validate(df.drop(columns=["Credit Score"]))
    bad = df.copy()
    bad.loc[0, TARGET] = "Extreme"
    with pytest.raises(SchemaError):
        validate(bad)


def test_noise_data_has_no_signal():
    # A permutation test is wrong 5% of the time by design, so the seed is fixed for a deterministic test.
    df = synthetic(signal=0.0, seed=3)
    X, y = split_xy(df)
    perm = audit.permutation_test(candidates()["Logistic Regression"], X, y, n_permutations=30)
    assert perm.p_value > 0.05
    assert abs(perm.score - 1 / 3) < 0.06


def test_real_signal_is_detected():
    df = synthetic(signal=1.0)
    X, y = split_xy(df)
    perm = audit.permutation_test(candidates()["Logistic Regression"], X, y, n_permutations=30)
    assert perm.p_value < 0.05
    assert perm.score > 0.5
    assoc = audit.association_tests(df)
    assert assoc.iloc[0]["feature"] == "Credit Score"
    assert assoc.iloc[0]["p_holm"] < 0.001


def test_positive_control_rises_with_strength():
    df = synthetic(signal=0.0)
    X, y = split_xy(df)
    model = candidates()["Logistic Regression"]
    weak = audit.positive_control(X, y, model, strength=0.25)
    strong = audit.positive_control(X, y, model, strength=1.0)
    assert strong > weak > 0.36


def test_majority_baseline_scores_exactly_chance():
    df = synthetic(signal=1.0)
    X, y = split_xy(df)
    scores = audit.cross_validated_scores(X, y, models={"m": candidates()["Always predict majority (baseline)"]})
    assert scores.loc[0, "balanced_accuracy"] == pytest.approx(1 / 3)
    assert scores.loc[0, "accuracy"] == pytest.approx((y == "Low").mean(), abs=0.01)


def test_verdict():
    perm = audit.PermutationResult(0.33, 0.33, 0.35, 0.6, np.array([0.33]))
    assoc = pd.DataFrame({"feature": ["a"], "p_holm": [0.3]})
    v = audit.verdict(perm, assoc)
    assert v["signal_detected"] is False and v["features_significant_after_correction"] == []
