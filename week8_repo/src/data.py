"""
src/data.py

Loads the raw triage CSV and turns it into a clean modelling table.
This is the Week 5 profiling and cleaning work, extracted into functions
that can be imported and tested, rather than run cell by cell.
"""

import pandas as pd
import numpy as np

# Clinically implausible cutoffs identified during Week 5 profiling.
# Values outside these ranges are treated as data entry errors, not
# genuine readings, and are replaced before imputation.
IMPLAUSIBLE_CUTOFFS = {
    "triage_vital_rr": 60,      # breaths per minute
    "triage_glucose": 800,      # mg/dL
}

VALID_ESI_LEVELS = [1, 2, 3, 4, 5]


def load_raw(path):
    """
    Read the raw triage CSV into a DataFrame.

    Parameters
    ----------
    path : str
        Path to yaleemmlc_admissionprediction_triage.csv

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(path, index_col=0)
    return df


def _replace_implausible_values(df):
    """Replace clinically impossible vital sign readings with NaN."""
    df = df.copy()
    for col, cutoff in IMPLAUSIBLE_CUTOFFS.items():
        if col in df.columns:
            mask = df[col] > cutoff
            df.loc[mask, col] = np.nan
    return df


def _impute_medians(df, columns):
    """Fill missing values in the given columns with the column median."""
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def clean(df):
    """
    Apply the full Week 5 cleaning pipeline to a raw triage DataFrame.

    Steps:
    1. Drop rows with no ESI label, or an ESI value outside 1 to 5
       (the target cannot be imputed, so these rows are excluded).
    2. Cast ESI to a clean integer (raw file stores it as float64).
    3. Replace clinically impossible vital sign readings with NaN.
    4. Impute the resulting gaps with the column median.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw triage data as returned by load_raw().

    Returns
    -------
    pandas.DataFrame
        Cleaned data, ready for feature selection.
    """
    df = df.copy()

    # Drop rows with no ESI label, or an out-of-range one. There is no
    # defensible way to impute the target itself, so these rows are
    # excluded rather than guessed at.
    df = df.dropna(subset=["esi"])
    df = df[df["esi"].isin(VALID_ESI_LEVELS)]
    df["esi"] = df["esi"].astype(int)

    df = _replace_implausible_values(df)
    df = _impute_medians(df, list(IMPLAUSIBLE_CUTOFFS.keys()))

    return df


def encode_gender(df):
    """
    Encode the gender column to a binary integer (1 = Male, 0 = Female),
    matching the convention used since Week 0.

    This is a general-purpose cleaning function. It is not applied by
    default in the modelling pipeline, since src/features.py excludes
    demographic columns, including gender, from the feature set by
    design (see the fairness note in docs/handover.md).
    """
    df = df.copy()
    gender_map = {"Male": 1, "Female": 0}
    df["gender"] = df["gender"].map(gender_map)
    return df


def validate_schema(df):
    """
    Confirm the cleaned DataFrame matches the contract the model expects.

    Raises AssertionError with a clear message if any check fails, so a
    silent data problem becomes a loud one instead. Used by clean() callers
    and by the schema test in tests/test_pipeline.py.
    """
    assert "esi" in df.columns, "esi column missing after cleaning"
    assert df["esi"].isin(VALID_ESI_LEVELS).all(), "esi contains a value outside 1 to 5"
    assert df["triage_vital_hr"].isna().sum() == 0, "triage_vital_hr still has gaps after cleaning"
    assert df["triage_vital_rr"].isna().sum() == 0, "triage_vital_rr still has gaps after cleaning"
    assert df["triage_glucose"].isna().sum() == 0, "triage_glucose still has gaps after cleaning"
    return True
