"""Entrenamiento multi-output (XGBoost + CatBoost) para Recuperacion de Cu y Mo.

Cada modelo predice simultaneamente [cu_recovery_pct, mo_recovery_pct]:
- XGBoost via `sklearn.multioutput.MultiOutputRegressor` (un XGBRegressor
  independiente por target internamente -- robusto entre versiones y deja
  sub-estimadores accesibles para SHAP por target).
- CatBoost nativo con `loss_function="MultiRMSE"`.

El ensamble de produccion es el promedio simple de ambos. La evaluacion
honesta de desempeño usa validacion cruzada temporal (`TimeSeriesSplit`):
entrena siempre en el pasado y evalua en el futuro, nunca al reves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor

from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

N_SPLITS = 5


@dataclass
class GeometallurgicalEnsemble:
    """Ensamble por promedio de XGBoost (multi-output) y CatBoost (MultiRMSE)."""

    xgb_model: MultiOutputRegressor
    catboost_model: CatBoostRegressor
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))

    def predict(self, X) -> np.ndarray:
        xgb_pred = np.asarray(self.xgb_model.predict(X))
        cb_pred = np.asarray(self.catboost_model.predict(X))
        return (xgb_pred + cb_pred) / 2.0


def _build_xgb() -> MultiOutputRegressor:
    base = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return MultiOutputRegressor(base)


def _build_catboost() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=350,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,  # evita que catboost escriba catboost_info/ en el cwd
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    metrics = {}
    for i, name in enumerate(target_names):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2}
    return metrics


def _aggregate_metrics(fold_metrics: list[dict], target_names: list[str]) -> dict:
    agg = {}
    for name in target_names:
        rmses = [f[name]["rmse"] for f in fold_metrics]
        maes = [f[name]["mae"] for f in fold_metrics]
        r2s = [f[name]["r2"] for f in fold_metrics]
        agg[name] = {
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        }
    return agg


def walk_forward_evaluate(
    X: np.ndarray, y: np.ndarray, target_names: list[str], n_splits: int = N_SPLITS
) -> dict[str, list[dict]]:
    """Validacion cruzada temporal (TimeSeriesSplit): entrena en el pasado, evalua en el futuro."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: dict[str, list[dict]] = {"xgboost": [], "catboost": [], "ensemble": []}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        xgb_model = _build_xgb()
        xgb_model.fit(X_train, y_train)
        xgb_pred = np.asarray(xgb_model.predict(X_test))

        cb_model = _build_catboost()
        cb_model.fit(X_train, y_train)
        cb_pred = np.asarray(cb_model.predict(X_test))

        ensemble_pred = (xgb_pred + cb_pred) / 2.0

        fold_metrics["xgboost"].append(_regression_metrics(y_test, xgb_pred, target_names))
        fold_metrics["catboost"].append(_regression_metrics(y_test, cb_pred, target_names))
        fold_metrics["ensemble"].append(_regression_metrics(y_test, ensemble_pred, target_names))

        ens = fold_metrics["ensemble"][-1]
        print(
            f"  [Fold {fold}/{n_splits}] n_train={len(train_idx)} n_test={len(test_idx)} | "
            f"Ensemble Cu R2={ens['cu_recovery_pct']['r2']:.3f} Mo R2={ens['mo_recovery_pct']['r2']:.3f}"
        )

    return fold_metrics


def fit_production_ensemble(X: np.ndarray, y: np.ndarray) -> GeometallurgicalEnsemble:
    """Ajusta el ensamble final sobre todos los datos disponibles (tras la evaluacion walk-forward)."""
    xgb_model = _build_xgb()
    xgb_model.fit(X, y)

    cb_model = _build_catboost()
    cb_model.fit(X, y)

    return GeometallurgicalEnsemble(xgb_model=xgb_model, catboost_model=cb_model)


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "block_model_flotation_features.parquet").sort("timestamp")
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df.select(TARGET_COLUMNS).to_numpy()

    print(f"Entrenando sobre {X.shape[0]} filas, {X.shape[1]} features, {y.shape[1]} targets")
    print(f"Validacion cruzada temporal (TimeSeriesSplit, {N_SPLITS} folds):")
    fold_metrics = walk_forward_evaluate(X, y, TARGET_COLUMNS)

    summary = {
        model_name: _aggregate_metrics(folds, TARGET_COLUMNS) for model_name, folds in fold_metrics.items()
    }

    print("\nResumen (promedio +- std entre folds):")
    for model_name, target_metrics in summary.items():
        for target, m in target_metrics.items():
            print(
                f"  [{model_name}] {target}: RMSE={m['rmse_mean']:.3f}+-{m['rmse_std']:.3f} "
                f"MAE={m['mae_mean']:.3f}+-{m['mae_std']:.3f} R2={m['r2_mean']:.3f}+-{m['r2_std']:.3f}"
            )

    print("\nAjustando ensamble de produccion sobre el dataset completo...")
    ensemble = fit_production_ensemble(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, MODELS_DIR / "geometallurgical_ensemble.joblib")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado en: {MODELS_DIR / 'geometallurgical_ensemble.joblib'}")
    print(f"Metricas guardadas en: {REPORTS_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
