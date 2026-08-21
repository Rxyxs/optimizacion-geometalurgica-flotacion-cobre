"""Feature engineering geometalurgico: ratios de mineralogia, interacciones
no lineales (dureza x mineralogia x reactivos) y features de perturbacion
de flujo de aire via transformada de Wavelet (PyWavelets).

La transformada de wavelet se aplica sobre la señal YA filtrada por Kalman
(`air_flow_m3_h_kf`): el objetivo es detectar perturbaciones reales del
proceso (paradas de soplador, bloqueos), no ruido de instrumento, que ya
fue removido en `wrangling.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pywt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

WAVELET = "db4"
WAVELET_LEVEL = 2

FEATURE_COLUMNS: list[str] = [
    # geologia
    "cu_grade_pct",
    "mo_grade_pct",
    "sgi_kwh_t",
    "pyrite_pct",
    "chalcopyrite_frac",
    "bornite_frac",
    # operacion (filtrada por Kalman)
    "ph_kf",
    "air_flow_m3_h_kf",
    "pct_solids_kf",
    "p80_um_kf",
    "collector_g_t",
    "frother_g_t",
    # mineralogia
    "cu_sulfide_selectivity",
    "chalco_bornite_ratio",
    # interacciones no lineales
    "sgi_pyrite_interaction",
    "sgi_p80_interaction",
    "collector_pyrite_interaction",
    "ph_pyrite_interaction",
    "collector_per_sulfide_unit",
    "sgi_squared",
    "p80_deviation_from_optimum",
    # perturbacion de aire (wavelet)
    "air_flow_perturbation_energy",
    "air_flow_perturbation_peak",
]

TARGET_COLUMNS = ["cu_recovery_pct", "mo_recovery_pct"]

P80_KNOWN_OPTIMUM_UM = 170.0


def add_mineralogy_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """Ratios de mineralogia: selectividad sulfuro-de-cobre/piritas y calcopirita/bornita."""
    return df.with_columns(
        [
            (
                (pl.col("chalcopyrite_frac") + pl.col("bornite_frac"))
                / (pl.col("pyrite_pct") / 100.0 + 0.05)
            ).alias("cu_sulfide_selectivity"),
            (pl.col("chalcopyrite_frac") / (pl.col("bornite_frac") + 1e-3)).alias("chalco_bornite_ratio"),
        ]
    )


def add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Interacciones no lineales entre dureza (SGI), mineralogia y reactivos."""
    total_sulfide_unit = pl.col("chalcopyrite_frac") + pl.col("bornite_frac") + pl.col("pyrite_pct") / 100.0
    return df.with_columns(
        [
            (pl.col("sgi_kwh_t") * pl.col("pyrite_pct")).alias("sgi_pyrite_interaction"),
            (pl.col("sgi_kwh_t") * pl.col("p80_um_kf")).alias("sgi_p80_interaction"),
            (pl.col("collector_g_t") * pl.col("pyrite_pct")).alias("collector_pyrite_interaction"),
            (pl.col("ph_kf") * pl.col("pyrite_pct")).alias("ph_pyrite_interaction"),
            (pl.col("collector_g_t") / (total_sulfide_unit + 1e-3)).alias("collector_per_sulfide_unit"),
            (pl.col("sgi_kwh_t") ** 2).alias("sgi_squared"),
            ((pl.col("p80_um_kf") - P80_KNOWN_OPTIMUM_UM) ** 2).alias("p80_deviation_from_optimum"),
        ]
    )


def _pad_to_multiple(signal: np.ndarray, factor: int) -> tuple[np.ndarray, int]:
    remainder = len(signal) % factor
    if remainder == 0:
        return signal, 0
    pad = factor - remainder
    return np.pad(signal, (0, pad), mode="edge"), pad


def add_air_flow_wavelet_features(
    df: pl.DataFrame, wavelet: str = WAVELET, level: int = WAVELET_LEVEL, window: int = 16
) -> pl.DataFrame:
    """Energia y pico de los coeficientes de detalle (SWT) del flujo de aire, por muestra.

    Usa la Transformada de Wavelet Estacionaria (`pywt.swt`), que preserva
    la longitud de la señal (a diferencia de la DWT clasica), asociando un
    valor de perturbacion a cada lectura individual. La energia se agrega
    en una ventana rodante para suavizar el indicador.
    """
    signal = df["air_flow_m3_h_kf"].to_numpy()
    padded, _ = _pad_to_multiple(signal, 2**level)

    coeffs = pywt.swt(padded, wavelet=wavelet, level=level, trim_approx=False)
    finest_detail = coeffs[-1][1][: len(signal)]  # (cA, cD) del nivel mas fino, recortado al largo original

    detail_series = pl.Series("detail", finest_detail)
    energy = (detail_series**2).rolling_mean(window_size=window, min_samples=1)
    peak = detail_series.abs().rolling_max(window_size=window, min_samples=1)

    return df.with_columns(
        [
            energy.alias("air_flow_perturbation_energy"),
            peak.alias("air_flow_perturbation_peak"),
        ]
    )


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Pipeline completo de feature engineering geometalurgico."""
    df = df.sort("timestamp")
    df = add_mineralogy_ratios(df)
    df = add_interaction_features(df)
    df = add_air_flow_wavelet_features(df)
    return df


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "block_model_flotation_clean.parquet")
    features = engineer_features(df)

    out_path = PROCESSED_DIR / "block_model_flotation_features.parquet"
    features.write_parquet(out_path)

    print(f"Filas: {features.height}")
    print(f"Columnas de features: {len(FEATURE_COLUMNS)}")
    null_counts = features.select(FEATURE_COLUMNS).null_count().sum_horizontal()[0]
    print(f"Nulos en columnas de features: {null_counts}")
    print(
        features.select(
            ["air_flow_perturbation_energy", "air_flow_perturbation_peak", "cu_sulfide_selectivity"]
        ).describe()
    )
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
