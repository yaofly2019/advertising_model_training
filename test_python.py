"""Unit tests for the advertising ROI training helpers."""

import importlib
from unittest.mock import Mock, patch

import pandas as pd
import pytest


with patch("google.cloud.storage.Client"):
    training = importlib.import_module("advertising_model_training")


@pytest.fixture
def campaign_data():
    return pd.DataFrame({
        "CAMPAIGN": ["a", "b", "c", "d", "e"],
        "CHANNEL": [
            "search_engine", "social_media", "video", "email", "search_engine"
        ],
        "DATE": [
            "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01",
            "not-a-date",
        ],
        "TOTAL_CLICKS": [10, 20, 30, 40, 50],
        "TOTAL_COST": [10.0, 20.0, 30.0, 40.0, 5.0],
        "ADS_SERVED": [100, 200, 300, 400, 500],
    })


def test_read_campaign_data_adds_year_and_month(campaign_data):
    with patch.object(training.pd, "read_csv", return_value=campaign_data):
        result = training.read_campaign_data("campaign.csv")

    assert result.loc[0, "YEAR"] == 2024
    assert result.loc[0, "MONTH"] == 1
    assert pd.isna(result.loc[4, "YEAR"])


def test_calculate_revenue_per_month_aggregates_rows():
    revenue_data = pd.DataFrame({
        "YEAR": [2024, 2024, 2024],
        "MONTH": [1, 1, 2],
        "REVENUE": [100.0, 50.0, 200.0],
    })
    with patch.object(training.pd, "read_csv", return_value=revenue_data):
        result = training.calculate_revenue_per_month("revenue.csv")

    actual = result[["YEAR", "MONTH", "REVENUE"]].to_dict("records")
    assert actual == [
        {"YEAR": 2024, "MONTH": 1, "REVENUE": 150.0},
        {"YEAR": 2024, "MONTH": 2, "REVENUE": 200.0},
    ]


def test_calculate_spend_per_month_pivots_channels(campaign_data):
    valid_campaigns = campaign_data.iloc[:4].copy()
    valid_campaigns["YEAR"] = 2024
    valid_campaigns["MONTH"] = 1

    result = training.calculate_spend_per_month(valid_campaigns)

    row = result.iloc[0]
    assert row["SEARCH_ENGINE"] == 10.0
    assert row["SOCIAL_MEDIA"] == 20.0
    assert row["VIDEO"] == 30.0
    assert row["EMAIL"] == 40.0


def test_merge_dataframes_drops_missing_rows_and_date_columns():
    revenue_data = pd.DataFrame({
        "YEAR": [2024, 2024],
        "MONTH": [1, 2],
        "REVENUE": [100.0, 200.0],
    })
    spend_data = pd.DataFrame({
        "YEAR": [2024, 2024],
        "MONTH": [1, 2],
        "SEARCH_ENGINE": [10.0, None],
        "SOCIAL_MEDIA": [20.0, 30.0],
        "VIDEO": [30.0, 40.0],
        "EMAIL": [40.0, 50.0],
    })

    result = training.merge_dataframes(revenue_data, spend_data)

    assert len(result) == 1
    assert "YEAR" not in result.columns
    assert "MONTH" not in result.columns
    assert result.iloc[0]["REVENUE"] == 100.0


def test_train_model_fits_grid_search():
    training_data = pd.DataFrame({
        "SEARCH_ENGINE": range(20),
        "SOCIAL_MEDIA": range(20, 40),
        "VIDEO": range(40, 60),
        "EMAIL": range(60, 80),
        "REVENUE": range(100, 120),
    })
    model = Mock()
    with patch.object(training, "GridSearchCV", return_value=model) as grid_search:
        result = training.train_model(training_data)

    model.fit.assert_called_once()
    grid_search.assert_called_once()
    assert grid_search.call_args.kwargs["cv"] == 10
    assert result[0] is model
    assert len(result) == 5


def test_evaluate_model_returns_train_and_test_scores():
    model = Mock()
    model.score.side_effect = [0.95, 0.90]
    X_train, y_train = Mock(), Mock()
    X_test, y_test = Mock(), Mock()

    scores = training.evaluate_model(model, X_train, y_train, X_test, y_test)

    assert scores == (0.95, 0.90)
    assert model.score.call_count == 2


def test_write_metrics_to_bigquery_inserts_serialized_metrics():
    client = Mock()
    metrics = {"r2_train": 0.95, "r2_test": 0.90}
    training_time = pd.Timestamp("2024-01-02 03:04:05").to_pydatetime()
    with (
        patch.object(training.bigquery, "Client", return_value=client),
        patch.object(training.bigquery, "Table", return_value="metrics-table"),
    ):
        training.write_metrics_to_bigquery("linear_regression", training_time, metrics)

    client.insert_rows_json.assert_called_once_with(
        "metrics-table",
        [{
            "algo_name": "linear_regression",
            "training_time": "2024-01-02 03:04:05",
            "model_metrics": '{"r2_train": 0.95, "r2_test": 0.9}',
        }],
    )


def test_save_model_dumps_and_uploads_artifact():
    model = Mock()
    blob = Mock()
    bucket = Mock()
    bucket.blob.return_value = blob

    with (
        patch.object(training, "dump") as dump,
        patch.object(training, "bucket", bucket),
    ):
        training.save_model(model)

    dump.assert_called_once_with(model, "model.joblib")
    bucket.blob.assert_called_once_with("advertising_roi/artifact/model.joblib")
    blob.upload_from_filename.assert_called_once_with("model.joblib")
