"""Pipeline maestro de extremo a extremo ("Single-Phase Master Pipeline").

Orquesta, en un solo comando y sin intervencion humana, las 8 etapas del
sistema de optimizacion geometalurgica de flotacion Cu-Mo:

    1. Ingesta y generacion sintetica de block model + telemetria de celda
    2. Limpieza: filtro de Kalman + imputacion espacial + Isolation Forest
    3. Feature engineering geometalurgico (mineralogia, interacciones, wavelets)
    4. Entrenamiento multi-output (XGBoost + CatBoost) con walk-forward CV
    5. Motor de optimizacion prescriptiva (Algoritmo Genetico)
    6. Explicabilidad SHAP + dashboard PDF/PNG + reporte bilingue (ES/EN)
    7. Simulacion forward-looking de planta (ley de cabeza, pH, reactivos)
       validada con ensambles walk-forward (TimeSeriesSplit, sin fuga)
    8. Optimizacion restringida via scipy.optimize (differential_evolution),
       validada cruzadamente contra el Algoritmo Genetico de la etapa 5

Cada etapa reutiliza el `main()` ya validado de su modulo correspondiente
(cada modulo tambien se puede ejecutar de forma independiente para
depuracion), leyendo/escribiendo sus artefactos en `data/` y `outputs/`.
Un error en cualquier etapa detiene el pipeline con un traceback claro de
en que etapa y por que fallo -- no hay reintentos silenciosos.

Ejecutar desde la raiz del repositorio con:
    python -m src.master_pipeline
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from src import (
    constrained_optimizer,
    data_generator,
    explainability,
    feature_engineering,
    modeling,
    optimizer,
    plant_simulation,
    wrangling,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("master_pipeline")

STAGES: list[tuple[str, callable]] = [
    ("1/8 Ingesta y generacion sintetica de block model + telemetria", data_generator.main),
    ("2/8 Limpieza: Kalman + imputacion espacial + Isolation Forest", wrangling.main),
    ("3/8 Feature engineering geometalurgico (mineralogia, interacciones, wavelets)", feature_engineering.main),
    ("4/8 Entrenamiento multi-output (XGBoost + CatBoost) con walk-forward CV", modeling.main),
    ("5/8 Motor de optimizacion prescriptiva (Algoritmo Genetico)", optimizer.main),
    ("6/8 Explicabilidad SHAP + dashboard + reporte bilingue (ES/EN)", explainability.main),
    ("7/8 Simulacion forward-looking de planta (walk-forward, sin fuga)", plant_simulation.main),
    ("8/8 Optimizacion restringida via scipy.optimize (vs. Algoritmo Genetico)", constrained_optimizer.main),
]


@contextmanager
def _stage(name: str):
    logger.info("-" * 70)
    logger.info(f"INICIO    {name}")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception(f"FALLO     {name}")
        raise
    else:
        logger.info(f"COMPLETO  {name} ({time.perf_counter() - start:.1f}s)")


def main() -> None:
    pipeline_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("PIPELINE MAESTRO -- Optimizacion Geometalurgica de Flotacion Cu-Mo")
    logger.info(f"{len(STAGES)} etapas, ejecucion continua sin intervencion humana")
    logger.info("=" * 70)

    for name, stage_fn in STAGES:
        with _stage(name):
            stage_fn()

    total = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETO en {total:.1f}s ({total / 60:.1f} min)")
    logger.info("Artefactos en: data/processed/, outputs/models/, outputs/reports/, outputs/plots/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
