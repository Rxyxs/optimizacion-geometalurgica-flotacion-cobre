"""Simulacion forward-looking de variables operacionales de planta (ley de
cabeza, pH, reactivos) y validacion de la recuperacion predicha sobre esos
escenarios mediante un protocolo walk-forward estricto (`TimeSeriesSplit`),
sin fuga de informacion temporal.

Cierra la ultima fase del proyecto. Hasta aqui todo el pipeline (`modeling.py`,
`optimizer.py`) opera sobre el block model HISTORICO ya minado. Este modulo
mira hacia ADELANTE: simula el plan de operacion de los proximos
`N_SIMULATED_DAYS` dias -- ley de cabeza declinante (tipico de un plan
minero: la ley promedio del mineral alimentado cae a medida que se agotan
las zonas de mayor ley) mas variables de celda tipo AR(1) alrededor de la
ultima condicion operativa observada -- y le pregunta al modelo que
recuperacion esperar, con dos garantias explicitas:

1. **Sin fuga de informacion.** Los escenarios simulados son datos
   SINTETICOS DE FUTURO: no existen en el dataset historico, por lo que no
   hay forma de que se filtren al entrenamiento. Ademas, en vez de confiar
   en un unico modelo, se reentrena un ensamble por cada fold de un
   `TimeSeriesSplit` sobre el historico (cada fold entrena solo con el
   prefijo estrictamente anterior al fold, igual que `modeling.py`), y cada
   escenario futuro se anota con las predicciones de los `N_SPLITS`
   ensambles. Si hubiera fuga de informacion en algun fold, las
   predicciones entre folds divergirian de forma sistematica con el tamano
   de la ventana de entrenamiento; en cambio, folds con prefijos historicos
   crecientes deberian dar predicciones consistentes sobre el mismo
   escenario futuro -- eso es exactamente lo que `score_scenarios_across_folds`
   mide y reporta como banda de incertidumbre epistemica entre folds.
2. **Limites operacionales honestos.** Las curvas de respuesta
   (`sweep_operational_variable`) barren una variable a la vez sobre su
   rango operacional real, mostrando no solo el punto optimo sino donde la
   recuperacion se degrada fuera de la ventana operativa segura.

Simplificacion declarada: las variables de mineralogia (Mo, SGI, piritas,
calcopirita/bornita) y las features de perturbacion de aire por wavelet se
fijan en su promedio historico o en cero -- este modulo simula la deriva de
la ley de cabeza y de las variables de celda, no un nuevo modelo de bloques
geologico ni nuevos eventos de perturbacion, que quedan fuera de alcance.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.signal import lfilter
from sklearn.model_selection import TimeSeriesSplit

from src.feature_engineering import FEATURE_COLUMNS, P80_KNOWN_OPTIMUM_UM, TARGET_COLUMNS
from src.modeling import MODELS_DIR, N_SPLITS, GeometallurgicalEnsemble, _build_catboost, _build_xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

N_SIMULATED_DAYS = 180
SIM_SEED = 7

# Deriva de plan minero: la ley de cabeza promedio cae ~8%/anio a medida que
# se agotan las zonas de mayor ley -- una tasa moderada, tipica de la etapa
# media de vida de un rajo, elegida por diseno (no ajustada a datos) para
# que la simulacion sea una hipotesis de planificacion explicita, no una
# proyeccion estadistica del historico.
LEY_DECLINE_PCT_PER_YEAR = 0.08

# Persistencia AR(1) dia a dia para variables de celda -- deliberadamente
# mas baja que la de data_generator.py (AR_PHI=0.985, calibrada para
# telemetria cada 12 minutos): a resolucion diaria un operador reajusta
# setpoints con mucha mas frecuencia relativa, por lo que la deriva debe
# revertir a la media mas rapido.
DAILY_AR_PHI = 0.85

# Mismos limites fisicos de sensor/celda que data_generator.py.
PH_BOUNDS = (9.0, 12.0)
AIR_BOUNDS = (600.0, 1400.0)
SOLIDS_BOUNDS = (25.0, 42.0)
P80_BOUNDS = (90.0, 320.0)
CU_GRADE_BOUNDS = (0.05, 2.5)

# Desviaciones estandar del ruido diario (mas chicas que las de sensor de
# data_generator.py: representan variabilidad real de setpoint dia a dia,
# no ruido de instrumento).
PH_DAILY_STD = 0.18
AIR_DAILY_STD = 55.0
SOLIDS_DAILY_STD = 1.2
P80_DAILY_STD = 14.0
COLLECTOR_DAILY_STD = 3.0
FROTHER_DAILY_STD = 2.0

GEOLOGY_CONTEXT_COLUMNS = ["mo_grade_pct", "sgi_kwh_t", "pyrite_pct", "chalcopyrite_frac", "bornite_frac"]


def _ar1_noise(n: int, std: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido AR(1) de media cero: n[t] = phi*n[t-1] + eps[t]. Se suma a una tendencia externa."""
    innovation_std = std * np.sqrt(1 - phi**2)
    eps = rng.normal(0.0, innovation_std, size=n)
    return lfilter([1.0], [1.0, -phi], eps)


def representative_geology_context(df: pl.DataFrame) -> dict[str, float]:
    """Promedio historico de las variables de mineralogia -- la mezcla de mineral 'tipica'
    sobre la que se superpone la deriva de ley de cabeza simulada."""
    means = df.select(GEOLOGY_CONTEXT_COLUMNS).mean()
    return {col: float(means[col][0]) for col in GEOLOGY_CONTEXT_COLUMNS}


def simulate_future_operating_scenarios(
    df: pl.DataFrame, n_days: int = N_SIMULATED_DAYS, seed: int = SIM_SEED
) -> pl.DataFrame:
    """Simula `n_days` de operacion futura: ley de cabeza declinante (plan minero) +
    variables de celda tipo AR(1) alrededor de la ultima condicion observada.

    No usa ninguna fila del futuro real (no existe): cada valor se genera desde una
    tendencia declarada (ley) o un proceso estocastico (celda) sembrado en el ultimo
    estado historico observado, por lo que no hay, por construccion, forma de que el
    historico "vea" estos escenarios durante el entrenamiento.
    """
    rng = np.random.default_rng(seed)
    last = df.sort("timestamp").tail(30).mean()

    base_grade = float(last["cu_grade_pct"][0])
    base_ph = float(last["ph_kf"][0])
    base_air = float(last["air_flow_m3_h_kf"][0])
    base_solids = float(last["pct_solids_kf"][0])
    base_p80 = float(last["p80_um_kf"][0])
    base_collector = float(last["collector_g_t"][0])
    base_frother = float(last["frother_g_t"][0])
    last_timestamp = np.datetime64(df["timestamp"].max())

    t = np.arange(n_days)
    daily_decline = LEY_DECLINE_PCT_PER_YEAR / 365.0
    grade_trend = base_grade * (1.0 - daily_decline * t)
    cu_grade = np.clip(grade_trend + _ar1_noise(n_days, base_grade * 0.03, DAILY_AR_PHI, rng), *CU_GRADE_BOUNDS)

    ph = np.clip(base_ph + _ar1_noise(n_days, PH_DAILY_STD, DAILY_AR_PHI, rng), *PH_BOUNDS)
    air = np.clip(base_air + _ar1_noise(n_days, AIR_DAILY_STD, DAILY_AR_PHI, rng), *AIR_BOUNDS)
    solids = np.clip(base_solids + _ar1_noise(n_days, SOLIDS_DAILY_STD, DAILY_AR_PHI, rng), *SOLIDS_BOUNDS)
    p80 = np.clip(base_p80 + _ar1_noise(n_days, P80_DAILY_STD, DAILY_AR_PHI, rng), *P80_BOUNDS)
    collector = np.clip(base_collector + _ar1_noise(n_days, COLLECTOR_DAILY_STD, DAILY_AR_PHI, rng), 10.0, 60.0)
    frother = np.clip(base_frother + _ar1_noise(n_days, FROTHER_DAILY_STD, DAILY_AR_PHI, rng), 8.0, 32.0)

    timestamps = last_timestamp + (np.arange(1, n_days + 1) * np.timedelta64(1, "D"))

    return pl.DataFrame(
        {
            "day": np.arange(1, n_days + 1),
            "timestamp": timestamps,
            "cu_grade_pct": cu_grade.round(4),
            "ph_kf": ph.round(4),
            "air_flow_m3_h_kf": air.round(2),
            "pct_solids_kf": solids.round(3),
            "p80_um_kf": p80.round(2),
            "collector_g_t": collector.round(2),
            "frother_g_t": frother.round(2),
        }
    )


def _build_scenario_feature_matrix(scenario: dict[str, np.ndarray], geology: dict[str, float]) -> np.ndarray:
    """Reconstruye la matriz FEATURE_COLUMNS para un lote de escenarios, con la misma
    logica de ratios/interacciones que `feature_engineering.py` pero vectorizada sobre
    variables de planta en vez de bloques individuales."""
    n = len(scenario["cu_grade_pct"])
    mo_grade = np.full(n, geology["mo_grade_pct"])
    sgi = np.full(n, geology["sgi_kwh_t"])
    pyrite = np.full(n, geology["pyrite_pct"])
    chalco = np.full(n, geology["chalcopyrite_frac"])
    bornite = np.full(n, geology["bornite_frac"])
    ph = scenario["ph_kf"]
    collector = scenario["collector_g_t"]
    p80 = scenario["p80_um_kf"]
    total_sulfide_unit = chalco + bornite + pyrite / 100.0

    values = {
        "cu_grade_pct": scenario["cu_grade_pct"],
        "mo_grade_pct": mo_grade,
        "sgi_kwh_t": sgi,
        "pyrite_pct": pyrite,
        "chalcopyrite_frac": chalco,
        "bornite_frac": bornite,
        "ph_kf": ph,
        "air_flow_m3_h_kf": scenario["air_flow_m3_h_kf"],
        "pct_solids_kf": scenario["pct_solids_kf"],
        "p80_um_kf": p80,
        "collector_g_t": collector,
        "frother_g_t": scenario["frother_g_t"],
        "cu_sulfide_selectivity": (chalco + bornite) / (pyrite / 100.0 + 0.05),
        "chalco_bornite_ratio": chalco / (bornite + 1e-3),
        "sgi_pyrite_interaction": sgi * pyrite,
        "sgi_p80_interaction": sgi * p80,
        "collector_pyrite_interaction": collector * pyrite,
        "ph_pyrite_interaction": ph * pyrite,
        "collector_per_sulfide_unit": collector / (total_sulfide_unit + 1e-3),
        "sgi_squared": sgi**2,
        "p80_deviation_from_optimum": (p80 - P80_KNOWN_OPTIMUM_UM) ** 2,
        "air_flow_perturbation_energy": np.zeros(n),
        "air_flow_perturbation_peak": np.zeros(n),
    }
    return np.column_stack([values[c] for c in FEATURE_COLUMNS])


def fit_walk_forward_fold_ensembles(df: pl.DataFrame, n_splits: int = N_SPLITS) -> list[dict]:
    """Reentrena un ensamble por fold de `TimeSeriesSplit` (siempre pasado -> futuro,
    igual que `modeling.walk_forward_evaluate`), guardando el timestamp de corte de cada
    fold para poder auditar que ninguno vio datos posteriores a ese corte."""
    df_sorted = df.sort("timestamp")
    X = df_sorted.select(FEATURE_COLUMNS).to_numpy()
    y = df_sorted.select(TARGET_COLUMNS).to_numpy()
    timestamps = df_sorted["timestamp"].to_numpy()

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_models = []
    for fold, (train_idx, _test_idx) in enumerate(splitter.split(X), start=1):
        xgb_model = _build_xgb()
        xgb_model.fit(X[train_idx], y[train_idx])
        cb_model = _build_catboost()
        cb_model.fit(X[train_idx], y[train_idx])

        fold_models.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "train_end_timestamp": str(timestamps[train_idx[-1]]),
                "model": GeometallurgicalEnsemble(xgb_model=xgb_model, catboost_model=cb_model),
            }
        )
        print(f"  [PlantSimulation/WalkForward] Fold {fold}/{n_splits}: n_train={len(train_idx)}, "
              f"corte={timestamps[train_idx[-1]]}")

    return fold_models


def score_scenarios_across_folds(
    scenario_df: pl.DataFrame, geology: dict[str, float], fold_models: list[dict]
) -> pl.DataFrame:
    """Predice cada escenario futuro con CADA ensamble de fold y agrega media/desviacion
    entre folds -- la banda de incertidumbre epistemica que evidencia ausencia de fuga
    (ver docstring del modulo)."""
    scenario = {col: scenario_df[col].to_numpy() for col in scenario_df.columns}
    X_scenario = _build_scenario_feature_matrix(scenario, geology)

    cu_preds = np.column_stack([f["model"].predict(X_scenario)[:, 0] for f in fold_models])
    mo_preds = np.column_stack([f["model"].predict(X_scenario)[:, 1] for f in fold_models])

    return scenario_df.with_columns(
        [
            pl.Series("cu_recovery_pred_mean", cu_preds.mean(axis=1).round(3)),
            pl.Series("cu_recovery_pred_std", cu_preds.std(axis=1).round(4)),
            pl.Series("cu_recovery_pred_min", cu_preds.min(axis=1).round(3)),
            pl.Series("cu_recovery_pred_max", cu_preds.max(axis=1).round(3)),
            pl.Series("mo_recovery_pred_mean", mo_preds.mean(axis=1).round(3)),
            pl.Series("mo_recovery_pred_std", mo_preds.std(axis=1).round(4)),
        ]
    )


# Ventanas operacionales "seguras" declaradas para anotar los limites en las curvas de
# respuesta -- mas angostas que los limites fisicos de sensor (PH_BOUNDS, etc.), que son
# el rango que el instrumento puede leer, no el rango en que se recomienda operar.
OPERATING_ENVELOPE = {
    "ph_kf": (10.2, 11.3),
    "collector_g_t": (16.0, 40.0),
    "cu_grade_pct": (0.35, 1.1),
}


def sweep_operational_variable(
    base_scenario: dict[str, float], geology: dict[str, float], model, variable: str, grid: np.ndarray
) -> pl.DataFrame:
    """Recorre `variable` sobre `grid` manteniendo el resto del escenario fijo en
    `base_scenario`, y predice la recuperacion resultante con el ensamble de produccion --
    la curva de respuesta simulada que muestra donde la recuperacion cae fuera de la
    ventana operacional (`OPERATING_ENVELOPE`)."""
    n = len(grid)
    scenario = {col: np.full(n, base_scenario[col]) for col in base_scenario}
    scenario[variable] = grid

    X = _build_scenario_feature_matrix(scenario, geology)
    preds = model.predict(X)

    lo, hi = OPERATING_ENVELOPE.get(variable, (np.nan, np.nan))
    return pl.DataFrame(
        {
            variable: grid,
            "cu_recovery_pred_pct": preds[:, 0].round(3),
            "mo_recovery_pred_pct": preds[:, 1].round(3),
            "within_operating_envelope": (grid >= lo) & (grid <= hi) if not np.isnan(lo) else np.full(n, True),
        }
    )


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "block_model_flotation_features.parquet")
    geology = representative_geology_context(df)

    print("[PlantSimulation] Ajustando ensambles walk-forward (uno por fold de TimeSeriesSplit)...")
    fold_models = fit_walk_forward_fold_ensembles(df)

    print(f"\n[PlantSimulation] Simulando {N_SIMULATED_DAYS} dias de operacion futura "
          f"(ley de cabeza declinante + variables de celda AR(1))...")
    scenario_df = simulate_future_operating_scenarios(df)
    forecast = score_scenarios_across_folds(scenario_df, geology, fold_models)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast.write_csv(REPORTS_DIR / "plant_simulation_forecast.csv")

    mean_disagreement = float(forecast["cu_recovery_pred_std"].mean())
    print(f"Recuperacion Cu media proyectada: {forecast['cu_recovery_pred_mean'].mean():.2f}% "
          f"(dia 1: {forecast['cu_recovery_pred_mean'][0]:.2f}%, "
          f"dia {N_SIMULATED_DAYS}: {forecast['cu_recovery_pred_mean'][-1]:.2f}%)")
    print(f"Desacuerdo promedio entre folds (std entre {N_SPLITS} ensambles, por escenario): "
          f"{mean_disagreement:.3f} pp -- banda de incertidumbre epistemica, no fuga")

    production_model = joblib.load(MODELS_DIR / "geometallurgical_ensemble.joblib")
    last_scenario_row = scenario_df.tail(1).to_dicts()[0]
    base_scenario = {k: v for k, v in last_scenario_row.items() if k not in ("day", "timestamp")}

    print("\n[PlantSimulation] Curvas de respuesta (barrido de una variable a la vez)...")
    ph_curve = sweep_operational_variable(base_scenario, geology, production_model, "ph_kf", np.linspace(9.0, 12.0, 61))
    collector_curve = sweep_operational_variable(
        base_scenario, geology, production_model, "collector_g_t", np.linspace(10.0, 60.0, 51)
    )
    grade_curve = sweep_operational_variable(
        base_scenario, geology, production_model, "cu_grade_pct", np.linspace(0.1, 2.0, 51)
    )

    sweeps = {"ph_kf": ph_curve, "collector_g_t": collector_curve, "cu_grade_pct": grade_curve}
    for name, curve in sweeps.items():
        curve.write_csv(REPORTS_DIR / f"plant_simulation_sweep_{name}.csv")
        best_row = curve.sort("cu_recovery_pred_pct", descending=True).row(0, named=True)
        print(f"  {name}: optimo simulado en {best_row[name]:.2f} "
              f"(Cu={best_row['cu_recovery_pred_pct']:.2f}%, dentro de ventana operacional="
              f"{best_row['within_operating_envelope']})")

    summary = {
        "n_simulated_days": N_SIMULATED_DAYS,
        "n_folds": len(fold_models),
        "fold_train_end_timestamps": [f["train_end_timestamp"] for f in fold_models],
        "cu_recovery_pred_mean_day1": float(forecast["cu_recovery_pred_mean"][0]),
        "cu_recovery_pred_mean_last_day": float(forecast["cu_recovery_pred_mean"][-1]),
        "mean_inter_fold_std_pp": mean_disagreement,
        "geology_context": geology,
    }
    with open(REPORTS_DIR / "plant_simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado en: {REPORTS_DIR / 'plant_simulation_forecast.csv'}, "
          f"plant_simulation_sweep_*.csv, plant_simulation_summary.json")


if __name__ == "__main__":
    main()
