"""Explicabilidad (SHAP), dashboard PDF/PNG y reporte bilingue (ES/EN).

SHAP se calcula sobre los sub-modelos XGBoost por target (`ensemble.xgb_model
.estimators_[i]`), ya que `MultiOutputRegressor` los mantiene como
regresores independientes y completamente compatibles con
`shap.TreeExplainer` -- mas robusto entre versiones que forzar SHAP sobre
CatBoost en modo MultiRMSE.

El reporte combina tres niveles de explicacion:
1. Importancia SHAP global por metal (Cu, Mo).
2. "Drivers de perdida de metal": entre los bloques de baja recuperacion,
   que features empujan la prediccion hacia abajo en promedio.
3. Explicaciones locales para los bloques que paso por el optimizador.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import shap

from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS
from src.modeling import GeometallurgicalEnsemble  # noqa: F401 -- requerido por joblib.load
from src.optimizer import RECOVERY_THRESHOLD_PCT

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"

SHAP_SAMPLE_SIZE = 2000
TOP_N_GLOBAL = 15
TOP_N_LOSS_DRIVERS = 10
TOP_N_LOCAL_FEATURES = 5
TOP_N_BLOCKS_FOR_LOCAL_EXPLANATION = 10

sns.set_theme(style="whitegrid")


def compute_shap_for_target(ensemble: GeometallurgicalEnsemble, X_sample: pd.DataFrame, target_idx: int):
    sub_model = ensemble.xgb_model.estimators_[target_idx]
    explainer = shap.TreeExplainer(sub_model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values


def global_importance_table(shap_values: np.ndarray, feature_names: list[str], top_n: int = TOP_N_GLOBAL) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def loss_driver_table(
    shap_values: np.ndarray, feature_names: list[str], is_low_recovery: np.ndarray, top_n: int = TOP_N_LOSS_DRIVERS
) -> pd.DataFrame:
    """Entre los bloques de baja recuperacion, que features empujan la prediccion hacia abajo."""
    low_shap = shap_values[is_low_recovery]
    if len(low_shap) == 0:
        return pd.DataFrame(columns=["feature", "mean_shap_contribution"])
    contribution = low_shap.mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_shap_contribution": contribution})
        .sort_values("mean_shap_contribution")
        .head(top_n)
        .reset_index(drop=True)
    )


def explain_block_local(shap_row: np.ndarray, feature_names: list[str], top_n: int = TOP_N_LOCAL_FEATURES) -> list[dict]:
    order = np.argsort(-np.abs(shap_row))[:top_n]
    return [
        {"feature": feature_names[i], "shap_value": round(float(shap_row[i]), 4)} for i in order
    ]


def save_dashboard(
    cu_importance: pd.DataFrame,
    mo_importance: pd.DataFrame,
    cu_loss_drivers: pd.DataFrame,
    recovery_series: pd.Series,
    uplift_series: pd.Series,
    metrics: dict,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Optimizacion Geometalurgica Cu-Mo — Dashboard de Resultados", fontsize=16, fontweight="bold")

    sns.barplot(data=cu_importance.iloc[::-1], x="mean_abs_shap", y="feature", ax=axes[0, 0], color="#B87333")
    axes[0, 0].set_title(f"Top features SHAP — Recuperacion Cu (R2={metrics['ensemble']['cu_recovery_pct']['r2_mean']:.2f})")
    axes[0, 0].set_xlabel("Mean |SHAP|")
    axes[0, 0].set_ylabel("")

    sns.barplot(data=mo_importance.iloc[::-1], x="mean_abs_shap", y="feature", ax=axes[0, 1], color="#6A8CAF")
    axes[0, 1].set_title(f"Top features SHAP — Recuperacion Mo (R2={metrics['ensemble']['mo_recovery_pct']['r2_mean']:.2f})")
    axes[0, 1].set_xlabel("Mean |SHAP|")
    axes[0, 1].set_ylabel("")

    sns.barplot(data=cu_loss_drivers.iloc[::-1], x="mean_shap_contribution", y="feature", ax=axes[0, 2], color="#C0392B")
    axes[0, 2].set_title("Drivers de perdida de metal (Cu, bloques bajo umbral)")
    axes[0, 2].set_xlabel("Contribucion SHAP promedio (negativa = perdida)")
    axes[0, 2].set_ylabel("")
    axes[0, 2].axvline(0, color="black", linewidth=0.8)

    sns.histplot(recovery_series, bins=50, ax=axes[1, 0], color="#2E86AB")
    axes[1, 0].axvline(RECOVERY_THRESHOLD_PCT, color="red", linestyle="--", label=f"Umbral {RECOVERY_THRESHOLD_PCT}%")
    axes[1, 0].set_title("Distribucion de recuperacion de Cu predicha (toda la flota)")
    axes[1, 0].set_xlabel("Recuperacion Cu predicha (%)")
    axes[1, 0].legend()

    sns.histplot(uplift_series, bins=25, ax=axes[1, 1], color="#27AE60")
    axes[1, 1].set_title("Uplift de recuperacion recomendado por el GA (bloques optimizados)")
    axes[1, 1].set_xlabel("Uplift de recuperacion Cu (puntos porcentuales)")

    metrics_text = "\n".join(
        f"{model}: Cu R2={m['cu_recovery_pct']['r2_mean']:.3f}  Mo R2={m['mo_recovery_pct']['r2_mean']:.3f}"
        for model, m in metrics.items()
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.8, "Resumen de metricas (walk-forward CV)", fontsize=12, fontweight="bold", transform=axes[1, 2].transAxes
    )
    axes[1, 2].text(0.0, 0.1, metrics_text, fontsize=10, family="monospace", transform=axes[1, 2].transAxes, va="bottom")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / "geometallurgical_dashboard.png"
    pdf_path = PLOTS_DIR / "geometallurgical_dashboard.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _build_report_text_es(
    metrics: dict, cu_importance: pd.DataFrame, cu_loss_drivers: pd.DataFrame, recommendations: pl.DataFrame
) -> str:
    top_feature = cu_importance.iloc[0]["feature"]
    top_loss_driver = cu_loss_drivers.iloc[0]["feature"] if len(cu_loss_drivers) else "N/D"
    n_opt = recommendations.height
    uplift_mean = recommendations["recovery_uplift_pct"].mean() if n_opt else 0.0
    r2_cu = metrics["ensemble"]["cu_recovery_pct"]["r2_mean"]
    r2_mo = metrics["ensemble"]["mo_recovery_pct"]["r2_mean"]

    return (
        "RESUMEN GEOMETALURGICO — OPTIMIZACION DE FLOTACION Cu-Mo\n"
        "=========================================================\n\n"
        f"El ensamble de produccion (XGBoost + CatBoost, promedio) explica un {r2_cu:.1%} de la "
        f"varianza de la recuperacion de Cu y un {r2_mo:.1%} de la de Mo en validacion cruzada "
        f"temporal (walk-forward), evaluado siempre sobre datos futuros no vistos en entrenamiento.\n\n"
        f"La variable con mayor influencia global sobre la recuperacion de Cu es '{top_feature}'. "
        f"Al enfocar el analisis SHAP solo en los bloques de baja recuperacion (bajo el umbral de "
        f"{RECOVERY_THRESHOLD_PCT}%), la variable que mas empuja la prediccion hacia abajo -- es decir, "
        f"el principal driver de perdida de metal -- es '{top_loss_driver}'.\n\n"
        f"El motor de optimizacion prescriptiva (Algoritmo Genetico) procesó {n_opt} bloques de baja "
        f"recuperacion y recomendo ajustes de pH, dosificacion de reactivos y P80 que en promedio "
        f"elevarian la recuperacion de Cu predicha en {uplift_mean:.2f} puntos porcentuales, sin exceder "
        f"el presupuesto de insumos definido.\n\n"
        "Recomendacion operativa: priorizar la revision de los bloques listados en "
        "'optimization_recommendations.csv', comenzando por los de mayor uplift potencial, y validar "
        "en planta los ajustes de pH y P80 antes de modificar la dosificacion de reactivos."
    )


def _build_report_text_en(
    metrics: dict, cu_importance: pd.DataFrame, cu_loss_drivers: pd.DataFrame, recommendations: pl.DataFrame
) -> str:
    top_feature = cu_importance.iloc[0]["feature"]
    top_loss_driver = cu_loss_drivers.iloc[0]["feature"] if len(cu_loss_drivers) else "N/A"
    n_opt = recommendations.height
    uplift_mean = recommendations["recovery_uplift_pct"].mean() if n_opt else 0.0
    r2_cu = metrics["ensemble"]["cu_recovery_pct"]["r2_mean"]
    r2_mo = metrics["ensemble"]["mo_recovery_pct"]["r2_mean"]

    return (
        "GEOMETALLURGICAL SUMMARY — Cu-Mo FLOTATION OPTIMIZATION\n"
        "=========================================================\n\n"
        f"The production ensemble (XGBoost + CatBoost, averaged) explains {r2_cu:.1%} of Cu recovery "
        f"variance and {r2_mo:.1%} of Mo recovery variance under temporal (walk-forward) cross-validation, "
        f"always evaluated on future, unseen data.\n\n"
        f"The single most influential variable for Cu recovery globally is '{top_feature}'. Restricting "
        f"the SHAP analysis to low-recovery blocks (below the {RECOVERY_THRESHOLD_PCT}% threshold), the "
        f"variable that pushes predictions down the most -- the primary metal-loss driver -- is "
        f"'{top_loss_driver}'.\n\n"
        f"The prescriptive optimization engine (Genetic Algorithm) processed {n_opt} low-recovery blocks "
        f"and recommended pH, reagent dosage, and P80 adjustments that would raise predicted Cu recovery "
        f"by {uplift_mean:.2f} percentage points on average, without exceeding the defined reagent budget.\n\n"
        "Operational recommendation: prioritize reviewing the blocks listed in "
        "'optimization_recommendations.csv', starting with the highest potential uplift, and validate "
        "the pH and P80 adjustments in-plant before modifying reagent dosing."
    )


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "block_model_flotation_features.parquet")
    ensemble: GeometallurgicalEnsemble = joblib.load(MODELS_DIR / "geometallurgical_ensemble.joblib")

    with open(REPORTS_DIR / "model_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    sample_df = df.sample(n=min(SHAP_SAMPLE_SIZE, df.height), seed=42)
    X_sample = sample_df.select(FEATURE_COLUMNS).to_pandas()

    print("Calculando SHAP para Cu...")
    cu_shap = compute_shap_for_target(ensemble, X_sample, target_idx=0)
    print("Calculando SHAP para Mo...")
    mo_shap = compute_shap_for_target(ensemble, X_sample, target_idx=1)

    cu_importance = global_importance_table(cu_shap, FEATURE_COLUMNS)
    mo_importance = global_importance_table(mo_shap, FEATURE_COLUMNS)

    cu_pred_sample = ensemble.predict(X_sample.to_numpy())[:, 0]
    is_low_recovery = cu_pred_sample < RECOVERY_THRESHOLD_PCT
    cu_loss_drivers = loss_driver_table(cu_shap, FEATURE_COLUMNS, is_low_recovery)

    print(f"\nTop 5 features SHAP (Cu): {cu_importance['feature'].head(5).tolist()}")
    print(f"Top 5 drivers de perdida de metal (Cu): {cu_loss_drivers['feature'].head(5).tolist()}")

    cu_importance.to_csv(REPORTS_DIR / "shap_importance_cu.csv", index=False)
    mo_importance.to_csv(REPORTS_DIR / "shap_importance_mo.csv", index=False)
    cu_loss_drivers.to_csv(REPORTS_DIR / "cu_metal_loss_drivers.csv", index=False)

    recommendations_path = REPORTS_DIR / "optimization_recommendations.parquet"
    recommendations = pl.read_parquet(recommendations_path) if recommendations_path.exists() else pl.DataFrame()

    local_explanations = []
    if recommendations.height > 0:
        top_blocks = recommendations.sort("recovery_uplift_pct", descending=True).head(
            TOP_N_BLOCKS_FOR_LOCAL_EXPLANATION
        )
        block_ids = set(top_blocks["block_id"].to_list())
        block_rows = df.filter(pl.col("block_id").is_in(block_ids))
        X_blocks = block_rows.select(FEATURE_COLUMNS).to_pandas()
        block_shap = compute_shap_for_target(ensemble, X_blocks, target_idx=0)

        for i, block_id in enumerate(block_rows["block_id"].to_list()):
            local_explanations.append(
                {"block_id": block_id, "top_contributing_features": explain_block_local(block_shap[i], FEATURE_COLUMNS)}
            )

    png_path, pdf_path = save_dashboard(
        cu_importance,
        mo_importance,
        cu_loss_drivers,
        recovery_series=df["cu_recovery_pct"].to_pandas(),
        uplift_series=recommendations["recovery_uplift_pct"].to_pandas() if recommendations.height else pd.Series([]),
        metrics=metrics,
    )
    print(f"\nDashboard guardado en: {png_path} / {pdf_path}")

    report = {
        "model_metrics": metrics,
        "shap_global_importance_cu": cu_importance.to_dict(orient="records"),
        "shap_global_importance_mo": mo_importance.to_dict(orient="records"),
        "cu_metal_loss_drivers": cu_loss_drivers.to_dict(orient="records"),
        "n_blocks_optimized": recommendations.height,
        "mean_recovery_uplift_pct": float(recommendations["recovery_uplift_pct"].mean()) if recommendations.height else None,
        "local_explanations_top_blocks": local_explanations,
    }
    with open(REPORTS_DIR / "geometallurgical_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    text_es = _build_report_text_es(metrics, cu_importance, cu_loss_drivers, recommendations)
    text_en = _build_report_text_en(metrics, cu_importance, cu_loss_drivers, recommendations)
    (REPORTS_DIR / "summary_es.txt").write_text(text_es, encoding="utf-8")
    (REPORTS_DIR / "summary_en.txt").write_text(text_en, encoding="utf-8")

    print(f"\nReporte JSON: {REPORTS_DIR / 'geometallurgical_report.json'}")
    print(f"Resumen ES: {REPORTS_DIR / 'summary_es.txt'}")
    print(f"Resumen EN: {REPORTS_DIR / 'summary_en.txt'}")


if __name__ == "__main__":
    main()
