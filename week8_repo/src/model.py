"""
src/model.py

Builds a classifier from a name and a dict of hyperparameters, and
evaluates it on six axes: accuracy, precision, recall, F1, training
time and inference time. This is the Weeks 6 and 7 modelling work,
extracted into functions that take explicit arguments rather than
relying on notebook-global variables.
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)

SUPPORTED_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

# Config keys can be more descriptive than the underlying scikit-learn
# family, for example "random_forest_high_recall" is still a
# RandomForestClassifier, just with different frozen hyperparameters.
# This maps any config key to the estimator family it should build.
MODEL_FAMILY = {
    "dummy": "dummy",
    "logistic_regression": "logistic_regression",
    "decision_tree": "decision_tree",
    "random_forest": "random_forest",
    "random_forest_high_recall": "random_forest",
    "gradient_boosting": "gradient_boosting",
}

# Models that need their features on a common scale. Others are trained
# on the raw feature values, matching the Week 6 and Week 7 notebooks.
NEEDS_SCALING = {"logistic_regression"}


def build_model(name, params, seed):
    """
    Construct an unfitted classifier from a config key and a
    hyperparameter dict.

    Parameters
    ----------
    name : str
        A key in MODEL_FAMILY, for example "random_forest_high_recall".
        Resolved to the underlying scikit-learn family before construction.
    params : dict
        Keyword arguments passed straight to the underlying scikit-learn
        constructor, for example {"n_estimators": 300, "max_depth": 10}.
    seed : int
        Random seed, passed as random_state to every model that accepts one.

    Returns
    -------
    A scikit-learn estimator, not yet fitted.
    """
    if name not in MODEL_FAMILY:
        raise ValueError(
            f"Unknown model name '{name}'. Supported models: {list(MODEL_FAMILY.keys())}"
        )
    family = MODEL_FAMILY[name]
    model_cls = SUPPORTED_MODELS[family]
    return model_cls(random_state=seed, **params)


def maybe_scale(X_train, X_test, name):
    """
    Fit a StandardScaler on X_train and apply it to both sets, but only
    for model families that need it (currently logistic regression).

    Returns (X_train_out, X_test_out, scaler_or_none).
    """
    family = MODEL_FAMILY.get(name, name)
    if family in NEEDS_SCALING:
        scaler = StandardScaler()
        X_train_out = scaler.fit_transform(X_train)
        X_test_out = scaler.transform(X_test)
        return X_train_out, X_test_out, scaler
    return X_train, X_test, None


def evaluate(model, X_test, y_test, primary_class=1):
    """
    Score a fitted model on the test set across accuracy, precision,
    recall, F1 (macro and weighted) and inference time.

    Parameters
    ----------
    model : a fitted scikit-learn estimator
    X_test, y_test : the held-out test set
    primary_class : the class label the project treats as the primary
        clinical metric, ESI 1 by default.

    Returns
    -------
    dict of metric name to value, plus 'predictions' and 'classification_report'.
    """
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    inference_time_per_row = (time.perf_counter() - t0) / len(X_test)

    recall_primary = recall_score(
        y_test, preds, labels=[primary_class], average=None, zero_division=0
    )[0]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "macro_precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        f"recall_class_{primary_class}": recall_primary,
        "inference_seconds_per_row": inference_time_per_row,
        "predictions": preds,
        "classification_report": classification_report(y_test, preds, digits=3),
    }
    return metrics


def fit_and_time(model, X_train, y_train):
    """
    Fit a model and return (fitted_model, training_time_seconds).
    Kept separate from evaluate() so training time and inference time
    are always measured the same way across every model in this project.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    return model, train_time
