"""
scripts/train.py

Single entry point for the triage pipeline. Reads config.yaml, wires
together src/data.py, src/features.py and src/model.py, and prints the
same six-axis evaluation used throughout Weeks 6 and 7.

Usage
-----
    python scripts/train.py --config config.yaml
    python scripts/train.py --config config.yaml --model random_forest_high_recall
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, features, model as model_module, utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the CariSurg triage model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model key from config.yaml's models section. Defaults to config's default_model.",
    )
    parser.add_argument(
        "--engineered-features", action="store_true",
        help="Turn on src.features.add_clinical_features(). Off by default; changes the frozen numbers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = utils.load_config(args.config)

    utils.set_seed(cfg["seed"])
    model_name = args.model or cfg["default_model"]

    print(f"Config loaded from {args.config}")
    print(f"Model: {model_name}")
    print(f"Seed: {cfg['seed']}")

    print("Loading and cleaning data...")
    df_raw = data.load_raw(cfg["data"]["raw_path"])
    df_clean = data.clean(df_raw)
    data.validate_schema(df_clean)
    print(f"Cleaned data shape: {df_clean.shape}")

    feature_cols = features.select_features(df_clean)
    X = df_clean[feature_cols]
    y = df_clean[cfg["data"]["target"]]

    X_train, X_test, y_train, y_test = utils.make_split(
        X, y, seed=cfg["seed"],
        test_size=cfg["split"]["test_size"],
        stratify=cfg["split"]["stratify"],
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    if args.engineered_features or cfg.get("engineered_features", {}).get("enabled", False):
        print("Engineered features ON: shock_index, hypoxia_flag, tachypnoea_flag added.")
        print("Note: this will not reproduce the frozen Week 6/7 numbers.")
        X_train = features.add_clinical_features(X_train)
        X_test = features.add_clinical_features(X_test)

    model_params = dict(cfg["models"][model_name])
    built_model = model_module.build_model(model_name, model_params, seed=cfg["seed"])

    X_train_final, X_test_final, scaler = model_module.maybe_scale(X_train, X_test, model_name)

    built_model, train_time = model_module.fit_and_time(built_model, X_train_final, y_train)
    print(f"Trained in {train_time:.2f} seconds")

    metrics = model_module.evaluate(built_model, X_test_final, y_test)

    print()
    print("=== Evaluation ===")
    print(f"Accuracy:            {metrics['accuracy']:.3f}")
    print(f"Macro precision:     {metrics['macro_precision']:.3f}")
    print(f"Macro recall:        {metrics['macro_recall']:.3f}")
    print(f"Macro F1:            {metrics['macro_f1']:.3f}")
    print(f"Weighted F1:         {metrics['weighted_f1']:.3f}")
    print(f"Recall on ESI 1:     {metrics['recall_class_1']:.3f}")
    print(f"Inference (ms/row):  {metrics['inference_seconds_per_row'] * 1000:.5f}")
    print()
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
