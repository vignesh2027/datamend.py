"""Shared fixtures for the datamend test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A well-formed DataFrame with no data quality issues."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, 200).astype(float),
            "income": rng.normal(55_000, 15_000, 200),
            "score": rng.uniform(0, 100, 200),
            "gender": rng.choice(["male", "female", "other"], 200),
            "city": rng.choice(["New York", "London", "Tokyo", "Paris"], 200),
        }
    )


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """A DataFrame with multiple data quality issues."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, 300).astype(float),
            "income": rng.normal(55_000, 15_000, 300),
            "score": rng.uniform(0, 100, 300),
            "gender": rng.choice(["male", "female", "Male", "MALE", "M", "Female"], 300),
            "city": rng.choice(["New York", "  London  ", "Tokyo", "Paris"], 300),
        }
    )
    # Inject nulls
    null_idx = rng.choice(df.index, 30, replace=False)
    df.loc[null_idx, "age"] = np.nan

    income_null_idx = rng.choice(df.index, 20, replace=False)
    df.loc[income_null_idx, "income"] = np.nan

    # Inject outliers
    df.loc[rng.choice(df.index, 5, replace=False), "income"] = 9_999_999.0

    # Inject duplicates
    dup_rows = df.iloc[:10].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


@pytest.fixture
def train_df() -> pd.DataFrame:
    """Training distribution DataFrame."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 500),
            "feature_b": rng.normal(5, 2, 500),
            "category": rng.choice(["A", "B", "C"], 500),
        }
    )


@pytest.fixture
def prod_stable_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Production DataFrame sampled from same distribution as train_df."""
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 300),
            "feature_b": rng.normal(5, 2, 300),
            "category": rng.choice(["A", "B", "C"], 300),
        }
    )


@pytest.fixture
def prod_drifted_df() -> pd.DataFrame:
    """Production DataFrame with significant distribution drift."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(3, 2, 300),   # Mean shifted by 3
            "feature_b": rng.normal(10, 4, 300),  # Mean and std shifted
            "category": rng.choice(["A", "D", "E"], 300),  # New categories
        }
    )


@pytest.fixture
def simple_sklearn_model(clean_df: pd.DataFrame) -> object:
    """A fitted sklearn RandomForestRegressor for testing FailureTrace."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor  # type: ignore

    X = clean_df[["age", "income", "score"]]
    y = clean_df["income"] * 0.01 + clean_df["age"] * 100
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model
