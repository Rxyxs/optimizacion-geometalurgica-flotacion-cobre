"""Generador sintetico de block model + telemetria de celdas de flotacion.

Simula una planta concentradora de cobre-molibdeno procesando bloques de un
modelo de bloques geologico. Cada fila es un bloque unico (coordenadas
espaciales x, y, banco) con sus atributos geometalurgicos (ley, dureza,
mineralogia) y las condiciones operacionales de la celda de flotacion con
las que fue procesado (pH, reactivos, P80, aire, % solidos), mas la
recuperacion metalurgica de Cu y Mo resultante.

Diseno de la simulacion:
- Las coordenadas espaciales se generan primero y se usa un
  `scipy.spatial.cKDTree` (build O(N log N), query vectorizado) para
  promediar cada bloque con sus vecinos y asi inducir continuidad
  geologica real (bloques cercanos tienen leyes/dureza similares).
- Las variables "sensor" de la celda (pH, aire, % solidos, P80) se generan
  como procesos AR(1) (media-reversion) via `scipy.signal.lfilter`, que
  representa el valor VERDADERO de planta -- sobre este valor se agrega
  ruido de medicion gaussiano para simular el instrumento real. La
  recuperacion responde al valor VERDADERO, no al ruidoso; esto hace que
  el filtro de Kalman de `wrangling.py` tenga un efecto medible y honesto
  sobre la calidad de las features.
- La recuperacion de Cu y Mo se calcula como una combinacion de funciones de
  respuesta no lineales (saturacion tipo isoterma de adsorcion para
  reactivos, optimo tipo campana para P80/aire/pH/solidos, mas
  interacciones geometalurgicas: piritas que consumen colector, dureza que
  penaliza liberacion) pasada por una transformacion logistica (sigmoide)
  que acota el resultado a un rango realista sin necesidad de recortar
  (clip) agresivamente la cola -- una suma lineal de terminos acotados en
  [0,1] con pesos grandes genera colas irrealmente extremas; la sigmoide
  comprime eso de forma suave y ademas dejar el ruido idiosincratico en un
  nivel moderado (no dominante sobre la señal) hace que los modelos de
  `modeling.py` puedan aprender relaciones genuinas (R2 > 0 de forma clara).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import lfilter
from scipy.spatial import cKDTree

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

N_BLOCKS_DEFAULT = 50_000
N_SPATIAL_NEIGHBORS = 8  # vecinos usados para suavizado geologico

# --- Rangos geometalurgicos base --------------------------------------
CU_GRADE_MEAN, CU_GRADE_STD = 0.62, 0.28  # % Cu
MO_GRADE_MEAN, MO_GRADE_STD = 0.018, 0.010  # % Mo
SGI_MEAN, SGI_STD = 14.0, 3.2  # indice de dureza, kWh/t (proxy Bond WI)
PYRITE_MEAN, PYRITE_STD = 3.0, 1.6  # % piritas (mineral de ganga)

# --- Rangos de operacion de celda (procesos AR(1)) ---------------------
AR_PHI = 0.985  # persistencia (mayor = deriva mas lenta, mas realista)

PH_MEAN, PH_STD, PH_NOISE = 10.6, 0.35, 0.12
AIR_MEAN, AIR_STD, AIR_NOISE = 950.0, 110.0, 35.0
SOLIDS_MEAN, SOLIDS_STD, SOLIDS_NOISE = 34.0, 2.4, 0.8
P80_MEAN, P80_STD, P80_NOISE = 180.0, 32.0, 9.0

COLLECTOR_BASE, COLLECTOR_NOISE = 22.0, 4.5  # g/t
FROTHER_BASE, FROTHER_STD = 18.0, 3.5  # g/t


def _ar1_process(n: int, mean: float, std: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Proceso AR(1) de media-reversion: x[t] = phi*x[t-1] + (1-phi)*mean + ruido[t].

    Implementado como un filtro IIR de primer orden (`scipy.signal.lfilter`)
    aplicado a la entrada `ruido[t] + (1-phi)*mean` -- el termino de deriva
    hacia la media debe sumarse en CADA paso (no solo en la condicion
    inicial), de lo contrario el proceso decae hacia 0 en vez de oscilar
    alrededor de `mean`.
    """
    noise = rng.normal(0.0, std * np.sqrt(1 - phi**2), size=n)
    drift = mean * (1 - phi)
    driven_input = noise + drift
    driven_input[0] = mean  # condicion inicial: arranca en la media
    return lfilter([1.0], [1.0, -phi], driven_input)


def _smooth_spatially(values: np.ndarray, tree: cKDTree, k: int = N_SPATIAL_NEIGHBORS) -> np.ndarray:
    """Promedia cada valor con sus k vecinos espaciales mas cercanos (continuidad geologica)."""
    _, neighbor_idx = tree.query(tree.data, k=k + 1)  # incluye el propio punto
    return values[neighbor_idx].mean(axis=1)


def _saturating(x: np.ndarray, k: float) -> np.ndarray:
    """Isoterma tipo Langmuir: respuesta con retornos decrecientes (cobertura superficial)."""
    return x / (x + k)


def _bell(x: np.ndarray, optimum: float, width: float) -> np.ndarray:
    """Curva de respuesta con optimo (campana gaussiana), para variables con punto dulce."""
    return np.exp(-((x - optimum) ** 2) / (2 * width**2))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_block_model_and_telemetry(
    n_blocks: int = N_BLOCKS_DEFAULT, seed: int = 42
) -> pl.DataFrame:
    """Genera el dataset combinado de block model + telemetria de flotacion + recuperacion.

    Devuelve la version "cruda con ruido de sensor" -- `wrangling.py` es
    responsable de limpiarla (Kalman, outliers, imputacion espacial).
    """
    rng = np.random.default_rng(seed)

    # --- Coordenadas espaciales (grilla de mina con jitter) ---
    side = int(np.ceil(np.sqrt(n_blocks)))
    xi, yi = np.meshgrid(np.arange(side), np.arange(side))
    x = (xi.ravel()[:n_blocks] * 25.0) + rng.normal(0, 3.0, n_blocks)
    y = (yi.ravel()[:n_blocks] * 25.0) + rng.normal(0, 3.0, n_blocks)
    bench = rng.integers(1, 16, n_blocks)  # 15 bancos
    z = 1600.0 - bench * 15.0

    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)

    # --- Geologia (con continuidad espacial via suavizado KDTree) ---
    cu_grade_raw = rng.normal(CU_GRADE_MEAN, CU_GRADE_STD, n_blocks)
    mo_grade_raw = rng.normal(MO_GRADE_MEAN, MO_GRADE_STD, n_blocks)
    sgi_raw = rng.normal(SGI_MEAN, SGI_STD, n_blocks)
    pyrite_raw = rng.normal(PYRITE_MEAN, PYRITE_STD, n_blocks)
    chalcopyrite_raw = rng.beta(6, 2, n_blocks)  # fraccion de sulfuros de Cu que es calcopirita

    cu_grade_pct = np.clip(_smooth_spatially(cu_grade_raw, tree), 0.05, 2.5)
    mo_grade_pct = np.clip(_smooth_spatially(mo_grade_raw, tree), 0.001, 0.08)
    sgi_kwh_t = np.clip(_smooth_spatially(sgi_raw, tree), 6.0, 24.0)
    pyrite_pct = np.clip(_smooth_spatially(pyrite_raw, tree), 0.2, 10.0)
    chalcopyrite_frac = np.clip(_smooth_spatially(chalcopyrite_raw, tree), 0.05, 0.98)
    bornite_frac = 1.0 - chalcopyrite_frac

    # --- Telemetria de celda: valor VERDADERO (AR1) + ruido de sensor ---
    ph_true = np.clip(_ar1_process(n_blocks, PH_MEAN, PH_STD, AR_PHI, rng), 9.0, 12.0)
    air_true = np.clip(_ar1_process(n_blocks, AIR_MEAN, AIR_STD, AR_PHI, rng), 600.0, 1400.0)
    solids_true = np.clip(_ar1_process(n_blocks, SOLIDS_MEAN, SOLIDS_STD, AR_PHI, rng), 25.0, 42.0)
    p80_true = np.clip(_ar1_process(n_blocks, P80_MEAN, P80_STD, AR_PHI, rng), 90.0, 320.0)

    # Eventos de perturbacion reales (paradas de soplador, bloqueos de linea):
    # pulsos cortos y decrecientes superpuestos al aire "verdadero". Le dan a
    # la deteccion por wavelets en feature_engineering.py señal real que
    # encontrar, no solo la deriva lenta del proceso AR(1).
    n_events = max(1, int(n_blocks * 0.004))
    event_starts = rng.choice(n_blocks - 6, size=n_events, replace=False)
    air_perturbation = np.zeros(n_blocks)
    for start in event_starts:
        duration = int(rng.integers(3, 7))
        magnitude = rng.normal(0, 220) * rng.choice([-1.0, 1.0])
        pulse = magnitude * np.exp(-np.arange(duration) / 2.0)
        end = min(start + duration, n_blocks)
        air_perturbation[start:end] += pulse[: end - start]
    air_true = np.clip(air_true + air_perturbation, 600.0, 1400.0)

    ph_sensor = ph_true + rng.normal(0, PH_NOISE, n_blocks)
    air_sensor = air_true + rng.normal(0, AIR_NOISE, n_blocks)
    solids_sensor = solids_true + rng.normal(0, SOLIDS_NOISE, n_blocks)
    p80_sensor = p80_true + rng.normal(0, P80_NOISE, n_blocks)

    # Reactivos: setpoints controlados por dosificacion (menos ruidosos),
    # con dosis de colector correlacionada al contenido total de sulfuros.
    total_sulfide = chalcopyrite_frac * 0.6 + bornite_frac * 0.6 + pyrite_pct / 100.0
    collector_g_t = np.clip(
        COLLECTOR_BASE + 18.0 * total_sulfide + rng.normal(0, COLLECTOR_NOISE, n_blocks), 10.0, 60.0
    )
    frother_g_t = np.clip(rng.normal(FROTHER_BASE, FROTHER_STD, n_blocks), 8.0, 32.0)

    # --- Funciones de respuesta metalurgica (sobre valores VERDADEROS) ---
    collector_stolen_by_pyrite = 0.8 * pyrite_pct
    effective_collector = np.clip(collector_g_t - collector_stolen_by_pyrite, 0.0, None)
    selectivity = 1.8 * pyrite_pct / (chalcopyrite_frac + bornite_frac + 0.05)
    hardness_penalty = sgi_kwh_t - SGI_MEAN  # centrado: 0 en el promedio de dureza

    def _centered(term: np.ndarray) -> np.ndarray:
        return term - term.mean()

    # Cu: pesos y anchos calibrados empiricamente (ver notebook de calibracion
    # en el README) para que, tras la sigmoide, la recuperacion de Cu caiga
    # ~86-87% en promedio con ~15-20% de bloques bajo el umbral de 82% que
    # usa el motor de optimizacion, y con una relacion señal/ruido que deja
    # a los modelos de `modeling.py` un techo de R2 ~0.6-0.7.
    cu_signal = (
        25.0 * _centered(_saturating(effective_collector, 14.0))
        + 10.0 * _centered(_saturating(frother_g_t, 10.0))
        + 18.0 * _centered(_bell(p80_true, optimum=170.0, width=45.0))
        + 12.0 * _centered(_bell(air_true, optimum=1000.0, width=180.0))
        + 15.0 * _centered(_bell(ph_true, optimum=10.8, width=0.7))
        + 10.0 * _centered(_bell(solids_true, optimum=34.0, width=5.0))
        - 0.4 * hardness_penalty
        - 1.0 * _centered(selectivity)
    )
    cu_raw = cu_signal + rng.normal(0, 3.0, n_blocks)
    cu_recovery_pct = 50.0 + (98.0 - 50.0) * _sigmoid((cu_raw + 9.6) / 6.5)

    # Mo: menos sensible a pH, optimo de P80 mas fino (mineral laminar),
    # rango de recuperacion tipico de flotacion bulk Cu-Mo (35-84%).
    mo_signal = (
        18.0 * _centered(_saturating(effective_collector, 20.0))
        + 16.0 * _centered(_saturating(frother_g_t, 12.0))
        + 16.0 * _centered(_bell(p80_true, optimum=140.0, width=50.0))
        + 10.0 * _centered(_bell(air_true, optimum=1050.0, width=210.0))
        + 6.0 * _centered(_bell(ph_true, optimum=10.2, width=1.3))
        - 0.3 * hardness_penalty
        - 0.6 * _centered(selectivity)
    )
    mo_raw = mo_signal + rng.normal(0, 3.0, n_blocks)
    mo_recovery_pct = 32.0 + (84.0 - 32.0) * _sigmoid((mo_raw + 3.0) / 8.0)

    timestamps = (
        np.datetime64("2024-01-01T00:00:00") + (np.arange(n_blocks) * np.timedelta64(12, "m"))
    ).astype("datetime64[us]")

    df = pl.DataFrame(
        {
            "block_id": [f"BLK-{i:06d}" for i in range(n_blocks)],
            "timestamp": timestamps,
            "x": x.round(2),
            "y": y.round(2),
            "bench": bench,
            "z": z,
            "cu_grade_pct": cu_grade_pct.round(4),
            "mo_grade_pct": mo_grade_pct.round(5),
            "sgi_kwh_t": sgi_kwh_t.round(3),
            "pyrite_pct": pyrite_pct.round(3),
            "chalcopyrite_frac": chalcopyrite_frac.round(4),
            "bornite_frac": bornite_frac.round(4),
            "ph_sensor": ph_sensor.round(4),
            "air_flow_m3_h_sensor": air_sensor.round(2),
            "pct_solids_sensor": solids_sensor.round(3),
            "p80_um_sensor": p80_sensor.round(2),
            "collector_g_t": collector_g_t.round(2),
            "frother_g_t": frother_g_t.round(2),
            "cu_recovery_pct": cu_recovery_pct.round(3),
            "mo_recovery_pct": mo_recovery_pct.round(3),
        }
    )

    # Nulifica ~2% de cada atributo geologico (mascaras independientes) para
    # simular datos de block-model incompletos y motivar la imputacion espacial.
    for col in ["cu_grade_pct", "sgi_kwh_t", "pyrite_pct"]:
        missing_mask = pl.Series(rng.random(n_blocks) < 0.02)
        df = df.with_columns(pl.when(missing_mask).then(None).otherwise(pl.col(col)).alias(col))

    return df


def save_raw_dataset(df: pl.DataFrame) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "block_model_flotation_raw.parquet"
    df.write_parquet(path)
    return path


def main() -> None:
    df = generate_block_model_and_telemetry()
    path = save_raw_dataset(df)

    print(f"Bloques generados: {df.height}")
    print(f"Nulos geologicos (simulan block-model incompleto): {df.null_count().sum_horizontal()[0]}")
    print(f"Cu recovery: mean={df['cu_recovery_pct'].mean():.2f}%  <82%: {(df['cu_recovery_pct'] < 82).sum()} bloques")
    print(f"Mo recovery: mean={df['mo_recovery_pct'].mean():.2f}%")
    print(f"Guardado en: {path}")


if __name__ == "__main__":
    main()
