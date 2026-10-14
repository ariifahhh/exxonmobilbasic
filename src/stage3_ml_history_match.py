"""Train and evaluate grid-feature models using the fixed 70/15/15 case split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "history_match_score"
IDENTIFIER_COLUMNS = {"case_id", "split", "source_file", "source_sheet"}


def candidate_models() -> dict[str, Pipeline]:
    """Small, deterministic model set appropriate for only 70 training cases."""
    return {
        "ridge_alpha_1": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "ridge_alpha_10": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=400, max_depth=4, min_samples_leaf=3, random_state=42, n_jobs=-1
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=400, max_depth=4, min_samples_leaf=3, random_state=42, n_jobs=-1
            )),
        ]),
    }


def score_predictions(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual = pd.Series(actual, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "spearman": float(actual.corr(pd.Series(predicted, index=actual.index), method="spearman")),
    }


def load_model_table(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(processed_dir / "grid_case_features.csv")
    scores = pd.read_csv(processed_dir / "case_history_match_ranking.csv")[["case_id", TARGET]]
    table = features.merge(scores, on="case_id", how="inner", validate="one_to_one")
    feature_columns = [column for column in table.columns if column not in IDENTIFIER_COLUMNS | {TARGET}]
    if table["split"].value_counts().to_dict() != {"train": 70, "validation": 15, "test": 15}:
        raise ValueError("Expected the fixed 70/15/15 split in grid_case_features.csv")
    return table, feature_columns


def plot_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    colors = {"validation": "#2f6db3", "test": "#e58d2c"}
    for split in ["validation", "test"]:
        subset = predictions.loc[predictions["split"] == split]
        axis.scatter(subset[TARGET], subset["predicted_score"], label=split.title(), alpha=0.8, color=colors[split])
    limits = [predictions[[TARGET, "predicted_score"]].min().min(), predictions[[TARGET, "predicted_score"]].max().max()]
    axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Observed curve-based history-match score")
    axis.set_ylabel("ML-predicted score")
    axis.set_title("Grid-feature ML diagnostic")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run(processed_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Choose a model on validation, then evaluate test once and export diagnostics."""
    table, feature_columns = load_model_table(processed_dir)
    train = table.loc[table["split"] == "train"].copy()
    validation = table.loc[table["split"] == "validation"].copy()
    test = table.loc[table["split"] == "test"].copy()
    evaluations = []

    for name, model in candidate_models().items():
        model.fit(train[feature_columns], train[TARGET])
        metrics = score_predictions(validation[TARGET], model.predict(validation[feature_columns]))
        evaluations.append({"model": name, "split": "validation", **metrics})
    comparison = pd.DataFrame(evaluations).sort_values(["mae", "rmse", "model"]).reset_index(drop=True)
    selected_name = comparison.iloc[0]["model"]
    selected_model = candidate_models()[selected_name]
    train_validation = pd.concat([train, validation], ignore_index=True)
    selected_model.fit(train_validation[feature_columns], train_validation[TARGET])

    prediction_rows = []
    for split, subset in [("validation", validation), ("test", test)]:
        model = selected_model if split == "test" else candidate_models()[selected_name].fit(train[feature_columns], train[TARGET])
        predicted = model.predict(subset[feature_columns])
        metrics = score_predictions(subset[TARGET], predicted)
        comparison = pd.concat([comparison, pd.DataFrame([{"model": selected_name, "split": split, **metrics}])], ignore_index=True)
        prediction_rows.append(pd.DataFrame({
            "case_id": subset["case_id"].to_numpy(),
            "split": split,
            TARGET: subset[TARGET].to_numpy(),
            "predicted_score": predicted,
            "selected_model": selected_name,
        }))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["predicted_rank_within_split"] = predictions.groupby("split")["predicted_score"].rank(method="first")
    test_metrics = comparison.loc[(comparison["model"] == selected_name) & (comparison["split"] == "test")].copy()
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": selected_model, "feature_columns": feature_columns, "selected_model": selected_name}, model_dir / "grid_history_match_model.joblib")
    comparison.to_csv(processed_dir / "ml_history_match_model_comparison.csv", index=False)
    predictions.to_csv(processed_dir / "ml_history_match_predictions.csv", index=False)
    test_metrics.to_csv(processed_dir / "ml_history_match_test_metrics.csv", index=False)
    plot_predictions(predictions, output_dir / "ml_history_match_diagnostic.png")
    return comparison, predictions, test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grid-feature ML diagnostics with a 70/15/15 split.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    comparison, _, test_metrics = run(args.processed_dir, args.output_dir)
    print(comparison.to_string(index=False))
    print("\nFinal one-time test metric:")
    print(test_metrics.to_string(index=False))
