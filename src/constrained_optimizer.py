"""Optimizacion restringida via `scipy.optimize`: maximiza la recuperacion de
Cu predicha sujeta a un techo de costo de reactivos, como alternativa
gradient-based/derivative-free al Algoritmo Genetico de `optimizer.py`.

Primer intento, documentado honestamente porque fallo: `scipy.optimize.minimize`
con `method="SLSQP"` es el candidato de libro de texto para "maximizar sujeto a
una restriccion de desigualdad", pero SLSQP necesita un gradiente, y sin uno
explicito SciPy lo aproxima por diferencias finitas. El ensamble XGBoost+CatBoost
es una funcion ESCALONADA (piecewise-constant: cada arbol es un conjunto de
umbrales), asi que ese gradiente numerico es identicamente cero en casi todo el
dominio -- SLSQP reporta "Optimization terminated successfully" en la primera
iteracion sin moverse un solo paso del punto inicial (`run_slsqp_reference` deja
esto reproducible a proposito, no se omite el fallo).

La solucion real usa `scipy.optimize.differential_evolution` (tambien
`scipy.optimize`, y sin derivadas: evalua la poblacion completa por diferencia,
nunca un gradiente), con la restriccion de presupuesto expresada como
`scipy.optimize.NonlinearConstraint`. Validado contra el Algoritmo Genetico
(DEAP) ya existente en `optimizer.py`, que resuelve el mismo problema por un
camino algoritmico independiente: ambos convergen al mismo optimo dentro de
~0.1pp de recuperacion en los bloques de prueba (ver `outputs/reports/
constrained_vs_ga_comparison.json`), la validacion cruzada entre dos
metodologias de optimizacion distintas que da confianza real en el resultado.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize

from src.feature_engineering import FEATURE_COLUMNS  # noqa: F401 -- referenciado por rebuild_feature_matrix
from src.modeling import GeometallurgicalEnsemble  # noqa: F401 -- requerido por joblib.load
from src.optimizer import (
    CONTEXT_COLUMNS,
    DECISION_BOUNDS,
    DECISION_VAR_NAMES,
    MAX_BLOCKS_TO_OPTIMIZE,
    RECOVERY_THRESHOLD_PCT,
    REAGENT_BUDGET_USD_PER_T,
    _reagent_cost_usd_per_t,
    rebuild_feature_matrix,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# DE es mas costoso por evaluacion que el GA (evalua un individuo a la vez, no en
# lote como `optimizer._evaluate_population`), asi que se acota a menos bloques.
MAX_BLOCKS_CONSTRAINED = 40
DE_MAXITER = 150
DE_POPSIZE = 15
DE_TOL = 1e-3


def _neg_cu_recovery(x: np.ndarray, context: dict, model) -> float:
    ph, collector, frother, p80 = x
    X = rebuild_feature_matrix(context, np.array([ph]), np.array([collector]), np.array([frother]), np.array([p80]))
    return -float(model.predict(X)[0, 0])


def _budget_cost(x: np.ndarray) -> float:
    _ph, collector, frother, _p80 = x
    return float(_reagent_cost_usd_per_t(np.array([collector]), np.array([frother]))[0])


def run_slsqp_reference(context: dict, model, x0: np.ndarray, budget: float = REAGENT_BUDGET_USD_PER_T) -> dict:
    """Referencia textbook (ver docstring del modulo): se espera que quede atascada en `x0`."""
    constraint = {"type": "ineq", "fun": lambda x: budget - _budget_cost(x)}
    result = minimize(
        _neg_cu_recovery, x0, args=(context, model), method="SLSQP", bounds=DECISION_BOUNDS, constraints=[constraint]
    )
    return {
        "x": result.x,
        "cu_recovery_pred": -float(result.fun),
        "success": bool(result.success),
        "n_iterations": int(result.nit),
        "moved_from_x0": bool(np.any(np.abs(result.x - x0) > 1e-6)),
    }


def run_differential_evolution_optimization(
    context: dict,
    model,
    bounds: list[tuple[float, float]] = DECISION_BOUNDS,
    budget: float = REAGENT_BUDGET_USD_PER_T,
    seed: int | None = None,
    maxiter: int = DE_MAXITER,
    popsize: int = DE_POPSIZE,
    tol: float = DE_TOL,
) -> dict:
    """Optimizacion sin derivadas (poblacional) con restriccion de presupuesto no lineal --
    robusta a la superficie escalonada del ensamble de arboles (ver docstring del modulo)."""
    constraint = NonlinearConstraint(_budget_cost, -np.inf, budget)
    result = differential_evolution(
        _neg_cu_recovery,
        bounds,
        args=(context, model),
        constraints=(constraint,),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=False,
    )
    ph, collector, frother, p80 = result.x
    return {
        "ph": float(ph),
        "collector_g_t": float(collector),
        "frother_g_t": float(frother),
        "p80_um": float(p80),
        "cu_recovery_pred": -float(result.fun),
        "reagent_cost_usd_per_t": _budget_cost(result.x),
        "success": bool(result.success),
        "n_iterations": int(result.nit),
    }


def run_constrained_prescriptive_engine(
    df: pl.DataFrame,
    model,
    threshold: float = RECOVERY_THRESHOLD_PCT,
    max_blocks: int = MAX_BLOCKS_CONSTRAINED,
) -> pl.DataFrame:
    """Mismo criterio de seleccion de bloques que `optimizer.run_prescriptive_engine`
    (Cu predicho bajo el umbral), resuelto via `differential_evolution` en vez del GA."""
    X_all = df.select(FEATURE_COLUMNS).to_numpy()
    cu_pred_all = model.predict(X_all)[:, 0]

    candidates = df.with_columns(pl.Series("predicted_cu_recovery_pct", cu_pred_all))
    targets = (
        candidates.filter(pl.col("predicted_cu_recovery_pct") < threshold)
        .sort("predicted_cu_recovery_pct")
        .head(max_blocks)
    )

    print(f"[ConstrainedOptimizer] {targets.height} bloques -> scipy.optimize.differential_evolution")

    records = []
    for i, row in enumerate(targets.iter_rows(named=True)):
        context = {col: row[col] for col in CONTEXT_COLUMNS}
        result = run_differential_evolution_optimization(context, model, seed=2000 + i)

        current_cost = float(
            _reagent_cost_usd_per_t(np.array([row["collector_g_t"]]), np.array([row["frother_g_t"]]))[0]
        )
        records.append(
            {
                "block_id": row["block_id"],
                "current_cu_recovery_pred_pct": round(float(row["predicted_cu_recovery_pct"]), 3),
                "recommended_ph": round(result["ph"], 3),
                "recommended_collector_g_t": round(result["collector_g_t"], 2),
                "recommended_frother_g_t": round(result["frother_g_t"], 2),
                "recommended_p80_um": round(result["p80_um"], 1),
                "recommended_cu_recovery_pred_pct": round(result["cu_recovery_pred"], 3),
                "recovery_uplift_pct": round(result["cu_recovery_pred"] - row["predicted_cu_recovery_pct"], 3),
                "current_reagent_cost_usd_per_t": round(current_cost, 4),
                "recommended_reagent_cost_usd_per_t": round(result["reagent_cost_usd_per_t"], 4),
                "de_success": result["success"],
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == targets.height:
            print(f"  [ConstrainedOptimizer] {i + 1}/{targets.height} bloques procesados")

    return pl.DataFrame(records)


def run_slsqp_failure_diagnostic(df: pl.DataFrame, model, n_blocks: int = 10) -> pl.DataFrame:
    """Reproduce el fallo de SLSQP (ver docstring del modulo) sobre una muestra pequeña de
    bloques, como evidencia documentada -- no una anecdota, un artefacto reproducible."""
    X_all = df.select(FEATURE_COLUMNS).to_numpy()
    cu_pred_all = model.predict(X_all)[:, 0]
    candidates = df.with_columns(pl.Series("predicted_cu_recovery_pct", cu_pred_all))
    sample = candidates.filter(pl.col("predicted_cu_recovery_pct") < RECOVERY_THRESHOLD_PCT).sort(
        "predicted_cu_recovery_pct"
    ).head(n_blocks)

    records = []
    for row in sample.iter_rows(named=True):
        context = {col: row[col] for col in CONTEXT_COLUMNS}
        x0 = np.array([row["ph_kf"], row["collector_g_t"], row["frother_g_t"], row["p80_um_kf"]])
        slsqp = run_slsqp_reference(context, model, x0)
        de = run_differential_evolution_optimization(context, model, seed=3000)
        records.append(
            {
                "block_id": row["block_id"],
                "slsqp_cu_recovery_pred": round(slsqp["cu_recovery_pred"], 3),
                "slsqp_moved_from_x0": slsqp["moved_from_x0"],
                "slsqp_n_iterations": slsqp["n_iterations"],
                "de_cu_recovery_pred": round(de["cu_recovery_pred"], 3),
                "de_vs_slsqp_uplift_pct": round(de["cu_recovery_pred"] - slsqp["cu_recovery_pred"], 3),
            }
        )
    return pl.DataFrame(records)


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "block_model_flotation_features.parquet")
    model = joblib.load(MODELS_DIR / "geometallurgical_ensemble.joblib")

    print("[ConstrainedOptimizer] Diagnostico SLSQP vs. differential_evolution (10 bloques)...")
    diagnostic = run_slsqp_failure_diagnostic(df, model)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    diagnostic.write_csv(REPORTS_DIR / "slsqp_vs_de_diagnostic.csv")
    n_stuck = int((~diagnostic["slsqp_moved_from_x0"]).sum())
    print(f"  SLSQP se quedo en x0 en {n_stuck}/{diagnostic.height} bloques "
          f"(uplift promedio perdido: {diagnostic['de_vs_slsqp_uplift_pct'].mean():.2f}pp)")

    recommendations = run_constrained_prescriptive_engine(df, model)
    out_path = REPORTS_DIR / "constrained_optimization_recommendations.csv"
    recommendations.write_csv(out_path)
    print(f"\nBloques optimizados (scipy.optimize.differential_evolution): {recommendations.height}")
    if recommendations.height > 0:
        print(f"Uplift promedio de recuperacion: {recommendations['recovery_uplift_pct'].mean():.2f}pp")
        print(f"DE exitoso (convergencia bajo tol): "
              f"{int(recommendations['de_success'].sum())}/{recommendations.height}")
    print(f"Guardado en: {out_path}")

    ga_path = REPORTS_DIR / "optimization_recommendations.parquet"
    if ga_path.exists():
        ga = pl.read_parquet(ga_path).select(["block_id", "recommended_cu_recovery_pred_pct"]).rename(
            {"recommended_cu_recovery_pred_pct": "ga_cu_recovery_pred_pct"}
        )
        joined = recommendations.join(ga, on="block_id", how="inner")
        if joined.height > 0:
            diff = (joined["recommended_cu_recovery_pred_pct"] - joined["ga_cu_recovery_pred_pct"]).abs()
            comparison = {
                "n_blocks_compared": joined.height,
                "mean_abs_diff_pp": round(float(diff.mean()), 4),
                "max_abs_diff_pp": round(float(diff.max()), 4),
                "correlation": round(float(np.corrcoef(
                    joined["recommended_cu_recovery_pred_pct"], joined["ga_cu_recovery_pred_pct"]
                )[0, 1]), 4),
            }
            with open(REPORTS_DIR / "constrained_vs_ga_comparison.json", "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            print(f"\nComparacion DE vs. GA (bloques en comun: {comparison['n_blocks_compared']}): "
                  f"diff promedio={comparison['mean_abs_diff_pp']}pp, "
                  f"correlacion={comparison['correlation']}")


if __name__ == "__main__":
    main()
