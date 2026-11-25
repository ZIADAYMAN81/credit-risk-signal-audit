"""Candidate models, all wrapped in leakage-safe pipelines (imputation and encoding are fitted per fold)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def preprocessor(numeric=NUMERIC, categorical=CATEGORICAL) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), list(numeric)),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), list(categorical)),
    ])


def candidates(seed: int = 0, numeric=NUMERIC, categorical=CATEGORICAL) -> dict[str, Pipeline]:
    """Two baselines that cannot learn anything, and three real models with class weighting."""
    def wrap(model):
        return Pipeline([("prep", preprocessor(numeric, categorical)), ("model", model)])

    return {
        "Always predict majority (baseline)": wrap(DummyClassifier(strategy="most_frequent")),
        "Random guess by class share (baseline)": wrap(DummyClassifier(strategy="stratified", random_state=seed)),
        "Logistic Regression": wrap(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": wrap(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                     n_jobs=-1, random_state=seed)),
        "Gradient Boosting": wrap(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                 class_weight="balanced", random_state=seed)),
    }
