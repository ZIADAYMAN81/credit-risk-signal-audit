"""Loading and validating the Financial Risk Assessment dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Risk Rating"
CLASSES = ["Low", "Medium", "High"]

NUMERIC = [
    "Age", "Income", "Credit Score", "Loan Amount", "Years at Current Job",
    "Debt-to-Income Ratio", "Assets Value", "Number of Dependents", "Previous Defaults",
    "Marital Status Change",
]
CATEGORICAL = [
    "Gender", "Education Level", "Marital Status", "Loan Purpose", "Employment Status",
    "Payment History",
]
# Location columns are tested for association but not used by the models:
# City is almost unique per row and Country/State are random pairings in this dataset.
LOCATION = ["City", "State", "Country"]
REQUIRED = NUMERIC + CATEGORICAL + LOCATION + [TARGET]


class SchemaError(ValueError):
    """Raised when the input file does not have the expected columns or labels."""


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    labels = set(df[TARGET].dropna().unique())
    unknown = labels - set(CLASSES)
    if unknown:
        raise SchemaError(f"unexpected risk ratings: {sorted(unknown)}")
    if df[TARGET].isna().any():
        raise SchemaError("the target column has missing values")
    return df


def load(path: str | Path) -> pd.DataFrame:
    """Read the CSV and check its schema. Missing feature values are kept (the models impute them)."""
    return validate(pd.read_csv(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[NUMERIC + CATEGORICAL], df[TARGET]
