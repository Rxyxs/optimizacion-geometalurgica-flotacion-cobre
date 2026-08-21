"""Motor de optimizacion prescriptiva: Algoritmo Genetico (DEAP) que
recomienda pH, dosificacion de reactivos y P80 para bloques con recuperacion
de Cu predicha por debajo del umbral de negocio, sin exceder un presupuesto
de insumos.

Diseno de rendimiento: DEAP evalua tradicionalmente un individuo a la vez,
lo que seria lento llamando `model.predict()` (XGBoost+CatBoost) miles de
veces. Aqui se usa el toolbox de DEAP para seleccion/cruce/mutacion, pero la
evaluacion de fitness se hace en **lote**: toda la poblacion de una
generacion se apila en una matriz y se predice de una sola vez, que es casi
tan rapido como predecir una sola fila para un ensamble de arboles.
"""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from deap import base, creator, tools

from src.feature_engineering import FEATURE_COLUMNS
from src.modeling import GeometallurgicalEnsemble  # noqa: F401 -- requerido por joblib.load

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# --- Economia de reactivos ---
COLLECTOR_PRICE_USD_PER_KG = 2.6
FROTHER_PRICE_USD_PER_KG = 3.9
REAGENT_BUDGET_USD_PER_T = 0.22
BUDGET_PENALTY_WEIGHT = 500.0  # puntos de recuperacion "castigados" por cada $/t sobre presupuesto

# --- Configuracion del motor prescriptivo ---
RECOVERY_THRESHOLD_PCT = 82.0
MAX_BLOCKS_TO_OPTIMIZE = 150  # acota el tiempo de ejecucion total del pipeline
POPULATION_SIZE = 40
N_GENERATIONS = 25
CXPB, MUTPB = 0.6, 0.3

DECISION_VAR_NAMES = ["ph", "collector_g_t", "frother_g_t", "p80_um"]
DECISION_BOUNDS = [(9.5, 11.5), (15.0, 55.0), (10.0, 30.0), (100.0, 300.0)]

CONTEXT_COLUMNS = [
    "cu_grade_pct",
    "mo_grade_pct",
    "sgi_kwh_t",
    "pyrite_pct",
    "chalcopyrite_frac",
    "bornite_frac",
    "air_flow_m3_h_kf",
    "pct_solids_kf",
    "cu_sulfide_selectivity",
    "chalco_bornite_ratio",
    "air_flow_perturbation_energy",
    "air_flow_perturbation_peak",
]

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def rebuild_feature_matrix(
    context: dict, ph: np.ndarray, collector: np.ndarray, frother: np.ndarray, p80: np.ndarray
) -> np.ndarray:
    """Reconstruye la matriz de features (mismo orden que FEATURE_COLUMNS) para un lote de
    candidatos del GA, recalculando toda columna derivada de las variables de decision."""
    n = len(ph)
    sgi = context["sgi_kwh_t"]
    pyrite = context["pyrite_pct"]
    chalco = context["chalcopyrite_frac"]
    bornite = context["bornite_frac"]
    total_sulfide_unit = chalco + bornite + pyrite / 100.0

    values = {
        "cu_grade_pct": np.full(n, context["cu_grade_pct"]),
        "mo_grade_pct": np.full(n, context["mo_grade_pct"]),
        "sgi_kwh_t": np.full(n, sgi),
        "pyrite_pct": np.full(n, pyrite),
        "chalcopyrite_frac": np.full(n, chalco),
        "bornite_frac": np.full(n, bornite),
        "ph_kf": ph,
        "air_flow_m3_h_kf": np.full(n, context["air_flow_m3_h_kf"]),
        "pct_solids_kf": np.full(n, context["pct_solids_kf"]),
        "p80_um_kf": p80,
        "collector_g_t": collector,
        "frother_g_t": frother,
        "cu_sulfide_selectivity": np.full(n, context["cu_sulfide_selectivity"]),
        "chalco_bornite_ratio": np.full(n, context["chalco_bornite_ratio"]),
        "sgi_pyrite_interaction": np.full(n, sgi * pyrite),
        "sgi_p80_interaction": sgi * p80,
        "collector_pyrite_interaction": collector * pyrite,
        "ph_pyrite_interaction": ph * pyrite,
        "collector_per_sulfide_unit": collector / (total_sulfide_unit + 1e-3),
        "sgi_squared": np.full(n, sgi**2),
        "p80_deviation_from_optimum": (p80 - 170.0) ** 2,
        "air_flow_perturbation_energy": np.full(n, context["air_flow_perturbation_energy"]),
        "air_flow_perturbation_peak": np.full(n, context["air_flow_perturbation_peak"]),
    }
    return np.column_stack([values[c] for c in FEATURE_COLUMNS])


def _reagent_cost_usd_per_t(collector: np.ndarray, frother: np.ndarray) -> np.ndarray:
    return collector * COLLECTOR_PRICE_USD_PER_KG / 1000.0 + frother * FROTHER_PRICE_USD_PER_KG / 1000.0


def _evaluate_population(population: np.ndarray, model, context: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evalua TODA la poblacion de una generacion en una sola llamada a `model.predict`."""
    ph, collector, frother, p80 = population[:, 0], population[:, 1], population[:, 2], population[:, 3]
    X = rebuild_feature_matrix(context, ph, collector, frother, p80)
    cu_pred = model.predict(X)[:, 0]

    cost = _reagent_cost_usd_per_t(collector, frother)
    over_budget = np.clip(cost - REAGENT_BUDGET_USD_PER_T, 0.0, None)
    fitness = cu_pred - BUDGET_PENALTY_WEIGHT * over_budget
    return fitness, cu_pred, cost


def _clip_individual(individual: list, bounds: list[tuple[float, float]]) -> list:
    for i, (lo, hi) in enumerate(bounds):
        individual[i] = min(max(individual[i], lo), hi)
    return individual


def run_genetic_algorithm(
    context: dict,
    model,
    bounds: list[tuple[float, float]] = DECISION_BOUNDS,
    population_size: int = POPULATION_SIZE,
    n_generations: int = N_GENERATIONS,
    seed: int | None = None,
) -> dict:
    """Busca, para un bloque, la combinacion (pH, colector, espumante, P80) que maximiza la
    recuperacion de Cu predicha sin exceder el presupuesto de reactivos."""
    rng_local = random.Random(seed)
    toolbox = base.Toolbox()

    def _init_individual():
        return creator.Individual([rng_local.uniform(lo, hi) for lo, hi in bounds])

    toolbox.register("individual", _init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register(
        "mutate", tools.mutGaussian, mu=0.0, sigma=[(hi - lo) * 0.12 for lo, hi in bounds], indpb=0.4
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    best_fitness = -np.inf
    best_individual: list | None = None
    best_info: dict = {}

    for _ in range(n_generations):
        pop_array = np.array(population, dtype=np.float64)
        fitness_vals, cu_preds, costs = _evaluate_population(pop_array, model, context)

        for ind, fit in zip(population, fitness_vals):
            ind.fitness.values = (float(fit),)

        gen_best_idx = int(np.argmax(fitness_vals))
        if fitness_vals[gen_best_idx] > best_fitness:
            best_fitness = fitness_vals[gen_best_idx]
            best_individual = list(population[gen_best_idx])
            best_info = {
                "cu_recovery_pred": float(cu_preds[gen_best_idx]),
                "reagent_cost_usd_per_t": float(costs[gen_best_idx]),
            }

        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, len(population))]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng_local.random() < CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mutant in offspring:
            if rng_local.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for ind in offspring:
            _clip_individual(ind, bounds)

        population = offspring

    assert best_individual is not None
    result = dict(zip(DECISION_VAR_NAMES, best_individual))
    result.update(best_info)
    return result


def run_prescriptive_engine(
    df: pl.DataFrame,
    model,
    threshold: float = RECOVERY_THRESHOLD_PCT,
    max_blocks: int = MAX_BLOCKS_TO_OPTIMIZE,
) -> pl.DataFrame:
    """Identifica bloques con Cu predicho bajo el umbral y corre el GA para los `max_blocks` peores."""
    X_all = df.select(FEATURE_COLUMNS).to_numpy()
    cu_pred_all = model.predict(X_all)[:, 0]

    candidates = df.with_columns(pl.Series("predicted_cu_recovery_pct", cu_pred_all))
    low_recovery = candidates.filter(pl.col("predicted_cu_recovery_pct") < threshold).sort(
        "predicted_cu_recovery_pct"
    )
    targets = low_recovery.head(max_blocks)

    print(
        f"[Optimizer] {low_recovery.height} bloques con Cu predicho < {threshold}% "
        f"-> optimizando los {targets.height} de menor recuperacion"
    )

    records = []
    for i, row in enumerate(targets.iter_rows(named=True)):
        context = {col: row[col] for col in CONTEXT_COLUMNS}
        result = run_genetic_algorithm(context, model, seed=1000 + i)

        current_cost = float(_reagent_cost_usd_per_t(np.array([row["collector_g_t"]]), np.array([row["frother_g_t"]]))[0])

        records.append(
            {
                "block_id": row["block_id"],
                "current_ph": round(row["ph_kf"], 3),
                "current_collector_g_t": round(row["collector_g_t"], 2),
                "current_frother_g_t": round(row["frother_g_t"], 2),
                "current_p80_um": round(row["p80_um_kf"], 1),
                "current_cu_recovery_pred_pct": round(float(row["predicted_cu_recovery_pct"]), 3),
                "recommended_ph": round(result["ph"], 3),
                "recommended_collector_g_t": round(result["collector_g_t"], 2),
                "recommended_frother_g_t": round(result["frother_g_t"], 2),
                "recommended_p80_um": round(result["p80_um"], 1),
                "recommended_cu_recovery_pred_pct": round(result["cu_recovery_pred"], 3),
                "recovery_uplift_pct": round(result["cu_recovery_pred"] - row["predicted_cu_recovery_pct"], 3),
                "current_reagent_cost_usd_per_t": round(current_cost, 4),
                "recommended_reagent_cost_usd_per_t": round(result["reagent_cost_usd_per_t"], 4),
            }
        )

        if (i + 1) % 25 == 0 or (i + 1) == targets.height:
            print(f"  [Optimizer] {i + 1}/{targets.height} bloques procesados")

    return pl.DataFrame(records)


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "block_model_flotation_features.parquet")
    ensemble = joblib.load(MODELS_DIR / "geometallurgical_ensemble.joblib")

    recommendations = run_prescriptive_engine(df, ensemble)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "optimization_recommendations.parquet"
    recommendations.write_parquet(out_path)
    recommendations.write_csv(REPORTS_DIR / "optimization_recommendations.csv")

    print(f"\nBloques optimizados: {recommendations.height}")
    if recommendations.height > 0:
        print(f"Uplift promedio de recuperacion: {recommendations['recovery_uplift_pct'].mean():.2f} pp")
        print(f"Costo de reactivos promedio (actual -> recomendado): "
              f"{recommendations['current_reagent_cost_usd_per_t'].mean():.4f} -> "
              f"{recommendations['recommended_reagent_cost_usd_per_t'].mean():.4f} USD/t")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
