"""
tests/test_pipeline.py

Two tests, matching the Week 8 brief:

1. test_clean_produces_valid_schema: after cleaning, is the data the
   shape and type the model expects? This is the data's contract with
   everything downstream of it.
2. test_smoke_train_predict: does the whole pipeline run on a tiny
   slice without crashing? This does not check accuracy, only that
   the pipeline holds together end to end.

A silent failure in a clinical data pipeline is the dangerous kind.
These tests exist to make a broken pipeline fail loudly instead.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src import data, features, model as model_module, utils

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "yaleemmlc_admissionprediction_triage.csv",
)


@pytest.fixture(scope="module")
def raw_df():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Raw data not found at {DATA_PATH}. Place the CSV in data/ to run this test.")
    return data.load_raw(DATA_PATH)


def test_clean_produces_valid_schema(raw_df):
    """The data's contract with the model: valid labels, no gaps, right types."""
    df = data.clean(raw_df)

    assert df["esi"].isin([1, 2, 3, 4, 5]).all(), "esi contains a value outside 1 to 5"
    assert df["esi"].dtype.kind in ("i", "u"), "esi should be a clean integer type after cleaning"
    assert df["triage_vital_hr"].isna().sum() == 0, "triage_vital_hr has gaps after cleaning"
    assert df["triage_vital_rr"].isna().sum() == 0, "triage_vital_rr has gaps after cleaning"
    assert df["triage_glucose"].isna().sum() == 0, "triage_glucose has gaps after cleaning"
    assert len(df) > 0, "clean() returned an empty DataFrame"

    # This also exercises the schema validator directly, so a change to
    # validate_schema() itself is covered by the same test run.
    assert data.validate_schema(df) is True


def test_smoke_train_predict(raw_df):
    """Runs the full pipeline on a small slice. Does not check accuracy,
    only that fit and predict complete and return the expected shape."""
    df = data.clean(raw_df)
    small_df = df.sample(n=min(200, len(df)), random_state=42)

    feature_cols = features.select_features(small_df)
    X = small_df[feature_cols]
    y = small_df["esi"]

    # A 200-row sample may not contain every ESI level, so this split is
    # not stratified, unlike the real pipeline, purely to keep the smoke
    # test fast and independent of class balance.
    X_train, X_test, y_train, y_test = utils.make_split(X, y, seed=42, stratify=False)

    built_model = model_module.build_model("decision_tree", {"max_depth": 3}, seed=42)
    built_model, _ = model_module.fit_and_time(built_model, X_train, y_train)

    preds = built_model.predict(X_test)
    assert len(preds) == len(y_test), "prediction count does not match test set size"
