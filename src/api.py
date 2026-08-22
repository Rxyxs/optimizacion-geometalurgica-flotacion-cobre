"""Servicio FastAPI de scoring y optimizacion en tiempo real.

Expone, sobre el ultimo dataset procesado por `master_pipeline.py`, el
scoring de Recuperacion Cu/Mo de cualquier bloque y ambos motores de
optimizacion prescriptiva (objetivo unico sobre Cu, y frente de Pareto
Cu-vs-Mo) bajo demanda -- sin tener que re-correr el pipeline completo.

Requiere haber corrido `python -m src.master_pipeline` (o al menos
`data_generator` + `wrangling` + `feature_engineering` + `modeling`)
al menos una vez, para que existan los artefactos que este servicio carga.

Requiere autenticacion por API key (header `X-API-Key`) en todos los
endpoints salvo `/health`, y aplica rate-limiting por IP.

Ejecutar desde la raiz del repositorio con:
    uvicorn src.api:app --reload

Variables de entorno:
    GEOMET_API_KEY     API key esperada (default: "dev-key-change-me").
    GEOMET_RATE_LIMIT  Limite por IP, formato de `slowapi` (default: "60/minute").
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.feature_engineering import FEATURE_COLUMNS
from src.modeling import GeometallurgicalEnsemble  # noqa: F401 -- requerido por joblib.load
from src.optimizer import (
    CONTEXT_COLUMNS,
    RECOVERY_THRESHOLD_PCT,
    run_genetic_algorithm,
    run_pareto_genetic_algorithm,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"

FEATURES_PATH = PROCESSED_DIR / "block_model_flotation_features.parquet"
MODEL_PATH = MODELS_DIR / "geometallurgical_ensemble.joblib"

API_KEY = os.environ.get("GEOMET_API_KEY", "dev-key-change-me")
RATE_LIMIT = os.environ.get("GEOMET_RATE_LIMIT", "60/minute")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Corre primero: python -m src.master_pipeline"
        )
    return path


features_df = pl.read_parquet(_require(FEATURES_PATH))
ensemble: GeometallurgicalEnsemble = joblib.load(_require(MODEL_PATH))

# Score de toda la flota calculado una vez al levantar el servicio (evita
# recorrer las 49k filas en cada request a /blocks/at-risk o /blocks/{id}/score).
_all_preds = ensemble.predict(features_df.select(FEATURE_COLUMNS).to_numpy())
ALL_SCORES = features_df.select(["block_id"]).with_columns(
    [
        pl.Series("cu_recovery_pred_pct", _all_preds[:, 0].round(3)),
        pl.Series("mo_recovery_pred_pct", _all_preds[:, 1].round(3)),
    ]
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="API key invalida o ausente. Envia el header 'X-API-Key'.",
        )


limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

app = FastAPI(
    title="Geometallurgical Flotation Optimization API",
    description=(
        "Scoring de Recuperacion Cu/Mo y recomendaciones de optimizacion (Algoritmo "
        "Genetico, objetivo unico y frente de Pareto) en tiempo real, sobre el ultimo "
        "dataset procesado por master_pipeline.py. Requiere API key (header 'X-API-Key') "
        "en todos los endpoints salvo /health."
    ),
    version="1.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


def _risk_level(cu_pred: float) -> str:
    if cu_pred < RECOVERY_THRESHOLD_PCT - 10:
        return "CRITICO"
    if cu_pred < RECOVERY_THRESHOLD_PCT:
        return "BAJO_UMBRAL"
    return "OK"


def _get_block_row(block_id: str) -> dict:
    match = features_df.filter(pl.col("block_id") == block_id)
    if match.height == 0:
        raise HTTPException(status_code=404, detail=f"Bloque '{block_id}' no encontrado.")
    return match.row(0, named=True)


def _block_context(row: dict) -> dict:
    return {col: row[col] for col in CONTEXT_COLUMNS}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_blocks": features_df.height}


@app.get("/blocks/at-risk", dependencies=[Depends(require_api_key)])
def list_at_risk_blocks(limit: int = 50) -> list[dict]:
    """Bloques con Cu predicho bajo el umbral de negocio, ordenados de peor a mejor."""
    at_risk = ALL_SCORES.filter(pl.col("cu_recovery_pred_pct") < RECOVERY_THRESHOLD_PCT).sort(
        "cu_recovery_pred_pct"
    )
    return at_risk.head(limit).to_dicts()


@app.get("/blocks/{block_id}/score", dependencies=[Depends(require_api_key)])
def score_block(block_id: str) -> dict:
    match = ALL_SCORES.filter(pl.col("block_id") == block_id)
    if match.height == 0:
        raise HTTPException(status_code=404, detail=f"Bloque '{block_id}' no encontrado.")
    row = match.row(0, named=True)
    return {
        "block_id": block_id,
        "cu_recovery_pred_pct": row["cu_recovery_pred_pct"],
        "mo_recovery_pred_pct": row["mo_recovery_pred_pct"],
        "risk_level": _risk_level(row["cu_recovery_pred_pct"]),
    }


@app.get("/blocks/{block_id}/optimize", dependencies=[Depends(require_api_key)])
def optimize_block(block_id: str) -> dict:
    """Algoritmo Genetico de objetivo unico: maximiza Cu sujeto al presupuesto de reactivos."""
    row = _get_block_row(block_id)
    X = np.array([[row[c] for c in FEATURE_COLUMNS]])
    current_cu = float(ensemble.predict(X)[0, 0])

    result = run_genetic_algorithm(_block_context(row), ensemble, seed=abs(hash(block_id)) % (2**31))

    return {
        "block_id": block_id,
        "current_cu_recovery_pred_pct": round(current_cu, 3),
        "recommended_ph": round(result["ph"], 3),
        "recommended_collector_g_t": round(result["collector_g_t"], 2),
        "recommended_frother_g_t": round(result["frother_g_t"], 2),
        "recommended_p80_um": round(result["p80_um"], 1),
        "recommended_cu_recovery_pred_pct": round(result["cu_recovery_pred"], 3),
        "recovery_uplift_pct": round(result["cu_recovery_pred"] - current_cu, 3),
        "reagent_cost_usd_per_t": round(result["reagent_cost_usd_per_t"], 4),
    }


@app.get("/blocks/{block_id}/optimize/pareto", dependencies=[Depends(require_api_key)])
def optimize_block_pareto(block_id: str) -> dict:
    """NSGA-II: frente de Pareto completo del trade-off Cu-vs-Mo para este bloque."""
    row = _get_block_row(block_id)
    solutions = run_pareto_genetic_algorithm(
        _block_context(row), ensemble, seed=abs(hash(block_id)) % (2**31)
    )

    return {
        "block_id": block_id,
        "n_solutions": len(solutions),
        "solutions": [
            {
                "ph": round(s["ph"], 3),
                "collector_g_t": round(s["collector_g_t"], 2),
                "frother_g_t": round(s["frother_g_t"], 2),
                "p80_um": round(s["p80_um"], 1),
                "cu_recovery_pred_pct": round(s["cu_recovery_pred"], 3),
                "mo_recovery_pred_pct": round(s["mo_recovery_pred"], 3),
                "reagent_cost_usd_per_t": round(s["reagent_cost_usd_per_t"], 4),
            }
            for s in solutions
        ],
    }
