"""Limpieza de datos: filtro de Kalman, imputacion espacial e Isolation Forest.

Orden de aplicacion (cada paso asume que el anterior ya corrio):

1. Filtro de Kalman escalar sobre las 4 variables de sensor de la celda
   (pH, aire, % solidos, P80) -- suaviza el ruido de medicion sin conocer
   los parametros verdaderos de la simulacion (a diferencia de
   `data_generator.py`, este modulo se comporta como lo haria un pipeline
   real que solo ve lecturas ruidosas).
2. Imputacion por lags espaciales: variables geologicas faltantes se
   completan con el promedio de sus k vecinos mas cercanos
   (`scipy.spatial.cKDTree`, consistente con el bloque de ingesta O(N log N)).
3. Deteccion y remocion de outliers multivariados con Isolation Forest.

Tambien incluye un diagnostico FFT rapido (frecuencia dominante de una
señal de sensor cruda) usado solo como QC informativo antes del filtrado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Canales de sensor ruidosos -> (columna_cruda, columna_filtrada, process_var, measurement_var)
KALMAN_CHANNELS: list[tuple[str, str, float, float]] = [
    ("ph_sensor", "ph_kf", 0.0020, 0.0150),
    ("air_flow_m3_h_sensor", "air_flow_m3_h_kf", 150.0, 1200.0),
    ("pct_solids_sensor", "pct_solids_kf", 0.08, 0.65),
    ("p80_um_sensor", "p80_um_kf", 8.0, 80.0),
]

SPATIAL_IMPUTE_COLUMNS = ["cu_grade_pct", "sgi_kwh_t", "pyrite_pct"]

ISOLATION_FOREST_FEATURES = [
    "cu_grade_pct",
    "mo_grade_pct",
    "sgi_kwh_t",
    "pyrite_pct",
    "chalcopyrite_frac",
    "ph_kf",
    "air_flow_m3_h_kf",
    "pct_solids_kf",
    "p80_um_kf",
    "collector_g_t",
    "frother_g_t",
    "cu_recovery_pct",
    "mo_recovery_pct",
]


def compute_fft_diagnostic(series: np.ndarray, label: str = "signal") -> dict:
    """Diagnostico FFT rapido: frecuencia y amplitud dominante (excluye componente DC)."""
    n = len(series)
    detrended = series - series.mean()
    spectrum = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n)
    magnitude = np.abs(spectrum)
    magnitude[0] = 0.0
    dominant_idx = int(np.argmax(magnitude))
    return {
        "signal": label,
        "dominant_frequency_cycles_per_sample": float(freqs[dominant_idx]),
        "dominant_amplitude": float(magnitude[dominant_idx]),
        "n_samples": n,
    }


def kalman_smooth_1d(measurements: np.ndarray, process_var: float, measurement_var: float) -> np.ndarray:
    """Filtro de Kalman escalar (modelo de caminata aleatoria) sobre una serie temporal."""
    n = len(measurements)
    smoothed = np.empty(n, dtype=np.float64)
    x = float(measurements[0])
    p = measurement_var
    smoothed[0] = x

    for t in range(1, n):
        p_pred = p + process_var
        k_gain = p_pred / (p_pred + measurement_var)
        x = x + k_gain * (float(measurements[t]) - x)
        p = (1.0 - k_gain) * p_pred
        smoothed[t] = x

    return smoothed


def apply_kalman_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica el filtro de Kalman a cada canal ruidoso definido en `KALMAN_CHANNELS`."""
    result = df
    for raw_col, filtered_col, process_var, measurement_var in KALMAN_CHANNELS:
        smoothed = kalman_smooth_1d(df[raw_col].to_numpy(), process_var, measurement_var)
        result = result.with_columns(pl.Series(filtered_col, smoothed))
    return result


def spatial_lag_impute(df: pl.DataFrame, columns: list[str], k: int = 8) -> pl.DataFrame:
    """Imputa nulos usando el promedio de los k vecinos espaciales mas cercanos no nulos."""
    coords = df.select(["x", "y", "z"]).to_numpy()
    tree = cKDTree(coords)
    _, neighbor_idx = tree.query(coords, k=k + 1)  # incluye el propio punto

    result = df
    for col in columns:
        values = df[col].to_numpy()
        is_null = np.isnan(values)
        if not is_null.any():
            continue

        neighbor_values = values[neighbor_idx]
        neighbor_is_null = np.isnan(neighbor_values)
        masked = np.where(neighbor_is_null, 0.0, neighbor_values)
        counts = np.clip((~neighbor_is_null).sum(axis=1), 1, None)
        neighbor_mean = masked.sum(axis=1) / counts

        imputed = np.where(is_null, neighbor_mean, values)
        result = result.with_columns(pl.Series(col, imputed))

    return result


def remove_outliers_isolation_forest(
    df: pl.DataFrame, feature_columns: list[str], contamination: float = 0.02, seed: int = 42
) -> tuple[pl.DataFrame, int]:
    """Detecta y remueve outliers multivariados. Devuelve (df_limpio, n_removidos)."""
    X = df.select(feature_columns).to_numpy()
    clf = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200, n_jobs=-1)
    labels = clf.fit_predict(X)  # 1 = inlier, -1 = outlier
    mask = labels == 1
    n_removed = int((~mask).sum())
    return df.filter(pl.Series(mask)), n_removed


def clean_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """Pipeline completo de limpieza: Kalman -> imputacion espacial -> Isolation Forest."""
    df = df.sort("timestamp")
    df = apply_kalman_filters(df)
    df = spatial_lag_impute(df, SPATIAL_IMPUTE_COLUMNS)
    df, _ = remove_outliers_isolation_forest(df, ISOLATION_FOREST_FEATURES)
    return df


def main() -> None:
    df = pl.read_parquet(RAW_DIR / "block_model_flotation_raw.parquet").sort("timestamp")

    fft_diag = compute_fft_diagnostic(df["ph_sensor"].to_numpy(), label="ph_sensor")
    print(
        f"[QC FFT] pH sensor -> frecuencia dominante: "
        f"{fft_diag['dominant_frequency_cycles_per_sample']:.5f} ciclos/muestra "
        f"(amplitud {fft_diag['dominant_amplitude']:.1f})"
    )

    df = apply_kalman_filters(df)
    for raw_col, filtered_col, _, _ in KALMAN_CHANNELS:
        raw_std = df[raw_col].std()
        filtered_std = df[filtered_col].std()
        print(f"[Kalman] {raw_col}: std {raw_std:.4f} -> {filtered_col}: std {filtered_std:.4f}")

    n_nulls_before = df.select(SPATIAL_IMPUTE_COLUMNS).null_count().sum_horizontal()[0]
    df = spatial_lag_impute(df, SPATIAL_IMPUTE_COLUMNS)
    n_nulls_after = df.select(SPATIAL_IMPUTE_COLUMNS).null_count().sum_horizontal()[0]
    print(f"[Imputacion espacial] nulos: {n_nulls_before} -> {n_nulls_after}")

    n_before = df.height
    df, n_removed = remove_outliers_isolation_forest(df, ISOLATION_FOREST_FEATURES)
    print(f"[Isolation Forest] {n_removed} bloques atipicos removidos ({n_removed / n_before:.2%})")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "block_model_flotation_clean.parquet"
    df.write_parquet(out_path)
    print(f"\nBloques finales: {df.height}")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
