"""
src/features.py

Turns cleaned columns into the clues the model actually trains on.
This is the Weeks 5 to 7 feature work: choosing which columns to use,
excluding leakage and demographics, and adding a couple of clinically
motivated derived features.
"""

import pandas as pd
import numpy as np

TARGET = "esi"

VITALS = [
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp", "triage_vital_rr",
    "triage_vital_o2", "triage_vital_temp", "triage_glucose",
]

# Known only after the visit ends. Excluded so the model never trains on
# information a nurse could not have had at the moment of triage.
LEAKAGE = ["disposition", "previousdispo"]

ADMIN = ["dep_name", "arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin"]

# Excluded from the feature set by design, not by omission. Age was found
# to correlate with ESI in Week 5, but the wider demographic block (race,
# ethnicity, insurance status, and so on) is excluded so the model cannot
# use protected characteristics as a shortcut, consistent with the equity
# concerns raised in the Week 5 feasibility memo and the Week 4 research
# proposal's risk register.
DEMOGRAPHICS = [
    "age", "gender", "ethnicity", "race", "lang", "religion",
    "maritalstatus", "employstatus", "insurance_status",
]


def select_features(df):
    """
    Return the list of columns used as model input.

    Chief complaint flags (cc_*) plus the seven vital signs. Everything
    else, the target, leakage columns, admin columns and demographic
    columns, is excluded explicitly rather than left in by default.

    Parameters
    ----------
    df : pandas.DataFrame
        A cleaned DataFrame, as returned by src.data.clean().

    Returns
    -------
    list of str
    """
    excluded = set([TARGET] + LEAKAGE + ADMIN + DEMOGRAPHICS)
    features = [c for c in df.columns if c not in excluded]
    return features


def add_clinical_features(X):
    """
    Add a small number of clinically motivated derived features.

    - shock_index: heart rate divided by systolic blood pressure. A value
      above 1.0 is an established informal marker of haemodynamic
      compromise (heart working harder than the blood pressure supports).
    - hypoxia_flag: 1 if oxygen saturation is below 92 percent, a level
      associated with clinically significant hypoxia.
    - tachypnoea_flag: 1 if respiratory rate is above 20 breaths per
      minute, the SIRS criterion used since Week 0.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix containing at least the vitals columns.

    Returns
    -------
    pandas.DataFrame
        A copy of X with the new columns added.
    """
    X = X.copy()

    if "triage_vital_hr" in X.columns and "triage_vital_sbp" in X.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            X["shock_index"] = X["triage_vital_hr"] / X["triage_vital_sbp"]
        X["shock_index"] = X["shock_index"].replace([np.inf, -np.inf], np.nan)
        X["shock_index"] = X["shock_index"].fillna(X["shock_index"].median())

    if "triage_vital_o2" in X.columns:
        X["hypoxia_flag"] = (X["triage_vital_o2"] < 92).astype(int)

    if "triage_vital_rr" in X.columns:
        X["tachypnoea_flag"] = (X["triage_vital_rr"] > 20).astype(int)

    return X


def encode_demographics(X, df):
    """
    One-hot encode the demographic columns and join them onto X.

    Off by default. This function exists so a reviewer or a future
    experiment can explicitly opt in to using demographic features, but
    it is never called by scripts/train.py without that being a deliberate,
    logged decision, since including them was ruled out on fairness grounds
    (see docs/handover.md, Known Limitations).

    Parameters
    ----------
    X : pandas.DataFrame
        The feature matrix to join onto.
    df : pandas.DataFrame
        The cleaned DataFrame containing the demographic columns.

    Returns
    -------
    pandas.DataFrame
    """
    demo_present = [c for c in DEMOGRAPHICS if c in df.columns]
    encoded = pd.get_dummies(df[demo_present], drop_first=True)
    return X.join(encoded)
