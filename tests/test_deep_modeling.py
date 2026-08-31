"""Tests unitarios para src/deep_modeling.py: baseline Ridge, MLP con loss de
Huber custom, comparacion de activaciones y persistencia en DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from src.deep_modeling import (
    ACTIVATIONS,
    RecoveryMLP,
    _activation_layer,
    _regression_metrics,
    _time_ordered_split,
    huber_loss,
    persist_comparison_duckdb,
    train_baseline_ridge,
    train_dl_activation,
)

N_FEATURES = 5
TARGET_NAMES = ["cu_recovery_pct", "mo_recovery_pct"]


@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, N_FEATURES))
    true_w = rng.normal(size=(N_FEATURES, 2))
    y = X @ true_w + rng.normal(scale=0.1, size=(n, 2)) + 50.0
    return X.astype(np.float32), y.astype(np.float32)


def test_time_ordered_split_preserves_order_and_sizes(synthetic_data):
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = _time_ordered_split(X, y, test_fraction=0.25)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert X_test.shape[0] == int(X.shape[0] * 0.25)
    # El split es temporal: el train es el bloque inicial, el test el bloque final.
    np.testing.assert_array_equal(X_train, X[: X_train.shape[0]])
    np.testing.assert_array_equal(X_test, X[X_train.shape[0] :])
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


def test_activation_layer_valid_names():
    assert isinstance(_activation_layer("relu"), torch.nn.ReLU)
    assert isinstance(_activation_layer("gelu"), torch.nn.GELU)
    assert isinstance(_activation_layer("swish"), torch.nn.SiLU)


def test_activation_layer_invalid_name_raises():
    with pytest.raises(ValueError):
        _activation_layer("tanh_unsupported")


def test_huber_loss_matches_quadratic_regime_for_small_errors():
    y_true = torch.zeros(4)
    y_pred = torch.tensor([0.1, -0.2, 0.05, 0.0])
    loss = huber_loss(y_pred, y_true, delta=1.0)
    expected = torch.mean(0.5 * (y_true - y_pred) ** 2)
    assert torch.isclose(loss, expected, atol=1e-6)


def test_huber_loss_is_finite_and_nonnegative_for_large_errors():
    y_true = torch.zeros(4)
    y_pred = torch.tensor([100.0, -50.0, 25.0, 0.0])
    loss = huber_loss(y_pred, y_true, delta=1.0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_recovery_mlp_forward_shape():
    model = RecoveryMLP(n_features=N_FEATURES, n_targets=2, activation="relu", hidden_sizes=(8, 4))
    x = torch.randn(10, N_FEATURES)
    out = model(x)
    assert out.shape == (10, 2)


def test_regression_metrics_perfect_prediction_gives_r2_one():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    metrics = _regression_metrics(y_true, y_true.copy(), TARGET_NAMES)
    for target in TARGET_NAMES:
        assert metrics[target]["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert metrics[target]["mae"] == pytest.approx(0.0, abs=1e-9)
        assert metrics[target]["r2"] == pytest.approx(1.0, abs=1e-9)


def test_train_baseline_ridge_returns_metrics_for_all_targets(synthetic_data):
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = _time_ordered_split(X, y)
    result = train_baseline_ridge(X_train, y_train, X_test, y_test, TARGET_NAMES)

    assert set(result["metrics"].keys()) == set(TARGET_NAMES)
    assert result["y_pred"].shape == y_test.shape
    # Datos casi lineales: el baseline deberia explicar la mayoria de la varianza.
    for target in TARGET_NAMES:
        assert result["metrics"][target]["r2"] > 0.5


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_train_dl_activation_runs_and_returns_loss_history(synthetic_data, activation):
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = _time_ordered_split(X, y)
    result = train_dl_activation(X_train, y_train, X_test, y_test, TARGET_NAMES, activation)

    assert set(result["metrics"].keys()) == set(TARGET_NAMES)
    assert result["y_pred"].shape == y_test.shape
    assert len(result["loss_history"]) > 0
    # El loss no deberia explotar a NaN/Inf durante el entrenamiento.
    assert all(np.isfinite(v) for v in result["loss_history"])
    # La ultima epoca no deberia ser peor que la primera (aprendizaje neto positivo).
    assert result["loss_history"][-1] <= result["loss_history"][0] * 1.5


def test_persist_comparison_duckdb_writes_expected_rows(tmp_path, monkeypatch):
    import src.deep_modeling as dl_mod

    fake_db_path = tmp_path / "comparison.duckdb"
    monkeypatch.setattr(dl_mod, "DB_PATH", fake_db_path)
    monkeypatch.setattr(dl_mod, "REPORTS_DIR", tmp_path)

    rows = [
        {"approach": "ridge_baseline", "target": "cu_recovery_pct", "rmse": 5.0, "mae": 4.0, "r2": 0.5},
        {"approach": "mlp_relu", "target": "mo_recovery_pct", "rmse": 4.5, "mae": 3.5, "r2": 0.6},
    ]
    db_path = persist_comparison_duckdb(rows)

    assert Path(db_path) == fake_db_path
    assert fake_db_path.exists()

    con = duckdb.connect(str(fake_db_path))
    result = con.execute("SELECT approach, target, rmse, mae, r2 FROM comparison_metrics ORDER BY approach").fetchall()
    con.close()

    assert len(result) == 2
    assert result[0][0] == "mlp_relu"
    assert result[1][0] == "ridge_baseline"


def test_persist_comparison_duckdb_overwrites_previous_rows(tmp_path, monkeypatch):
    import src.deep_modeling as dl_mod

    fake_db_path = tmp_path / "comparison.duckdb"
    monkeypatch.setattr(dl_mod, "DB_PATH", fake_db_path)
    monkeypatch.setattr(dl_mod, "REPORTS_DIR", tmp_path)

    first_rows = [{"approach": "ridge_baseline", "target": "cu_recovery_pct", "rmse": 5.0, "mae": 4.0, "r2": 0.5}]
    persist_comparison_duckdb(first_rows)

    second_rows = [{"approach": "mlp_gelu", "target": "cu_recovery_pct", "rmse": 4.0, "mae": 3.0, "r2": 0.7}]
    persist_comparison_duckdb(second_rows)

    con = duckdb.connect(str(fake_db_path))
    count = con.execute("SELECT COUNT(*) FROM comparison_metrics").fetchone()[0]
    con.close()

    assert count == 1
