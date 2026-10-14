import pandas as pd

from src.stage3_extract_grid_features import split_for_case
from src.stage3_ml_history_match import score_predictions


def test_fixed_case_split():
    assert split_for_case(1) == "train"
    assert split_for_case(70) == "train"
    assert split_for_case(71) == "validation"
    assert split_for_case(85) == "validation"
    assert split_for_case(86) == "test"
    assert split_for_case(100) == "test"


def test_prediction_metrics_perfect_prediction():
    metrics = score_predictions(pd.Series([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["spearman"] == 1
