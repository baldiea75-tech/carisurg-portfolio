"""
src/utils.py

Small shared helpers used across the pipeline: config loading, seeding,
and a train/test split wrapper that always uses the project's
stratification and reproducibility rules.
"""

import random
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def load_config(path):
    """
    Read a YAML config file into a plain Python dict.

    Parameters
    ----------
    path : str
        Path to config.yaml.

    Returns
    -------
    dict
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """
    Seed Python's random module and NumPy, so any stochastic step outside
    scikit-learn's own random_state arguments is also reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)


def make_split(X, y, seed, test_size=0.2, stratify=True):
    """
    Wrap train_test_split with the project's standard settings: an
    80/20 split, stratified on the target by default, using the given
    seed. Centralising this in one place means every script and notebook
    that calls it produces the same split, rather than each one having
    its own slightly different train_test_split call.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    seed : int
    test_size : float
    stratify : bool
        If True, stratify on y. Set False only for debugging.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, stratify=strat, random_state=seed)
