"""Comparacion de 3 enfoques de modelado para Recuperacion de Cu y Mo:

    (a) Baseline interpretable -- Ridge multi-output sobre features estandarizadas.
    (b) Ensamble de arboles -- ya cubierto por `src/modeling.py` (XGBoost + CatBoost),
        cuyas metricas walk-forward se leen desde `outputs/reports/model_metrics.json`.
    (c) Deep learning -- MLP en PyTorch con loss custom (Huber suavizado) y barrido
        de activaciones ReLU / GELU / Swish (SiLU).

Reutiliza las mismas features/targets que `src/modeling.py` (FEATURE_COLUMNS,
TARGET_COLUMNS) y el mismo split temporal (holdout final ordenado por tiempo,
sin fuga). Persiste la tabla comparativa en DuckDB y en JSON, y genera graficos
de curvas de loss por activacion y predicho-vs-real / residuos.

Ejecutar de forma independiente con:
    python -m src.deep_modeling
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import polars as pl
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
DB_PATH = REPORTS_DIR / "model_comparison.duckdb"

TEST_FRACTION = 0.2
HIDDEN_SIZES = (64, 32)
N_EPOCHS = 60
LEARNING_RATE = 1e-3
HUBER_DELTA = 1.0
ACTIVATIONS = ("relu", "gelu", "swish")
RANDOM_SEED = 42

COLOR_CU = "#B87333"
COLOR_MO = "#6A8CAF"
COLOR_LOSS = ["#B87333", "#6A8CAF", "#27AE60"]


def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU()  # Swish == SiLU (x * sigmoid(x))
    raise ValueError(f"Activacion desconocida: {name}")


class RecoveryMLP(nn.Module):
    """MLP simple: features geometalurgicas -> [Cu, Mo recovery %]."""

    def __init__(self, n_features: int, n_targets: int, activation: str, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_activation_layer(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    """Loss de Huber implementada a mano (custom loss): cuadratica cerca de 0,
    lineal lejos -- robusta a outliers de recuperacion sin perder sensibilidad
    fina cerca del optimo, a diferencia de MSE puro."""
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)


def _time_ordered_split(X: np.ndarray, y: np.ndarray, test_fraction: float = TEST_FRACTION):
    n = X.shape[0]
    n_test = int(n * test_fraction)
    n_train = n - n_test
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    metrics = {}
    for i, name in enumerate(target_names):
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    return metrics


def train_baseline_ridge(X_train, y_train, X_test, y_test, target_names: list[str]) -> dict:
    """(a) Baseline interpretable: Ridge lineal multi-output sobre features estandarizadas."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    return {
        "metrics": _regression_metrics(y_test, y_pred, target_names),
        "y_pred": y_pred,
        "model": model,
        "scaler": scaler,
    }


def train_dl_activation(
    X_train, y_train, X_test, y_test, target_names: list[str], activation: str, seed: int = RANDOM_SEED
) -> dict:
    """(c) Entrena la MLP con una activacion dada y loss de Huber custom; devuelve
    metricas de test, historial de loss por epoca y predicciones para graficar."""
    torch.manual_seed(seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    y_train_s = (y_train - y_mean) / y_std

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

    model = RecoveryMLP(n_features=X_train.shape[1], n_targets=y_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history: list[float] = []
    model.train()
    for _epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        y_pred_t = model(X_train_t)
        loss = huber_loss(y_pred_t, y_train_t)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        y_pred_s = model(X_test_t).numpy()
    y_pred = y_pred_s * y_std + y_mean  # des-estandarizar

    return {
        "metrics": _regression_metrics(y_test, y_pred, target_names),
        "y_pred": y_pred,
        "loss_history": loss_history,
        "model": model,
    }


def _load_ensemble_metrics() -> dict | None:
    path = REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    ensemble = summary.get("ensemble")
    if ensemble is None:
        return None
    return {
        target: {"rmse": m["rmse_mean"], "mae": m["mae_mean"], "r2": m["r2_mean"]} for target, m in ensemble.items()
    }


def persist_comparison_duckdb(rows: list[dict]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS comparison_metrics (
            approach VARCHAR,
            target VARCHAR,
            rmse DOUBLE,
            mae DOUBLE,
            r2 DOUBLE,
            run_ts TIMESTAMP DEFAULT current_timestamp
        )
        """
    )
    con.execute("DELETE FROM comparison_metrics")
    con.executemany(
        "INSERT INTO comparison_metrics (approach, target, rmse, mae, r2) VALUES (?, ?, ?, ?, ?)",
        [(r["approach"], r["target"], r["rmse"], r["mae"], r["r2"]) for r in rows],
    )
    con.close()
    return DB_PATH


def save_loss_curves_plot(activation_histories: dict[str, list[float]]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for (activation, history), color in zip(activation_histories.items(), COLOR_LOSS):
        ax.plot(range(1, len(history) + 1), history, label=activation.upper(), color=color, linewidth=2)
    ax.set_title("MLP -- Curva de loss (Huber custom) por activacion", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss de Huber (entrenamiento)")
    ax.legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "dl_activation_loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_loss_curves_animation(activation_histories: dict[str, list[float]], max_frames: int = 60) -> Path:
    """Version animada ('racing line chart') de `save_loss_curves_plot`: cada linea de
    loss se dibuja progresivamente frame a frame, con una etiqueta flotante en la punta
    de cada curva mostrando la activacion y el valor de loss actual. Usa el mismo
    `loss_history` real (Huber, por epoca) ya calculado en el entrenamiento -- sin
    datos inventados."""
    activations = list(activation_histories.keys())
    n_epochs = len(next(iter(activation_histories.values())))
    n_frames = min(max_frames, n_epochs)
    # Indices de epoca (reales) a revelar en cada frame, submuestreando si hay mas
    # epocas que frames deseados.
    frame_epoch_idx = np.unique(
        np.linspace(1, n_epochs, n_frames, dtype=int)
    )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    all_losses = np.concatenate([np.array(h) for h in activation_histories.values()])
    ax.set_xlim(1, n_epochs)
    ax.set_ylim(0, float(all_losses.max()) * 1.1)
    ax.set_title("MLP -- Curva de loss (Huber custom) por activacion", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss de Huber (entrenamiento)")

    lines = {}
    labels = {}
    for activation, color in zip(activations, COLOR_LOSS):
        (line,) = ax.plot([], [], label=activation.upper(), color=color, linewidth=2)
        lines[activation] = line
        labels[activation] = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9,
            color="white",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="white", alpha=0.85),
        )
    ax.legend(loc="upper right")
    fig.tight_layout()

    def _update(frame_idx: int):
        up_to = int(frame_epoch_idx[frame_idx])
        for activation, history in activation_histories.items():
            x = np.arange(1, up_to + 1)
            y = np.array(history[:up_to])
            lines[activation].set_data(x, y)
            current_val = history[up_to - 1]
            labels[activation].xy = (up_to, current_val)
            labels[activation].set_position((10, 0))
            labels[activation].set_text(f"{activation.upper()}: {current_val:.3f}")
        return list(lines.values()) + list(labels.values())

    ani = FuncAnimation(fig, _update, frames=len(frame_epoch_idx), interval=150, blit=False)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "dl_activation_loss_curves_animated.gif"
    ani.save(path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")
    return path


def save_predicted_vs_actual_plot(y_test: np.ndarray, y_pred: np.ndarray, target_names: list[str], best_activation: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"MLP ({best_activation.upper()}) -- Predicho vs. Real y Residuos", fontsize=14, fontweight="bold")
    colors = [COLOR_CU, COLOR_MO]

    for i, (name, color) in enumerate(zip(target_names, colors)):
        ax = axes[0, i]
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.3, s=10, color=color)
        lo, hi = float(y_test[:, i].min()), float(y_test[:, i].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name} -- predicho vs. real")
        ax.set_xlabel("Real (%)")
        ax.set_ylabel("Predicho (%)")

        residuals = y_test[:, i] - y_pred[:, i]
        ax_res = axes[1, i]
        ax_res.scatter(y_pred[:, i], residuals, alpha=0.3, s=10, color=color)
        ax_res.axhline(0, color="black", linewidth=0.8)
        ax_res.set_title(f"{name} -- residuos")
        ax_res.set_xlabel("Predicho (%)")
        ax_res.set_ylabel("Residuo (real - predicho)")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "dl_predicted_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "block_model_flotation_features.parquet").sort("timestamp")
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df.select(TARGET_COLUMNS).to_numpy()
    X_train, X_test, y_train, y_test = _time_ordered_split(X, y)

    print(f"Split temporal: train={X_train.shape[0]} test={X_test.shape[0]}")

    print("\n(a) Entrenando baseline interpretable (Ridge)...")
    baseline_result = train_baseline_ridge(X_train, y_train, X_test, y_test, TARGET_COLUMNS)
    for target, m in baseline_result["metrics"].items():
        print(f"  [ridge_baseline] {target}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} R2={m['r2']:.3f}")

    print("\n(c) Entrenando MLP con barrido de activaciones (loss de Huber custom)...")
    dl_results: dict[str, dict] = {}
    for activation in ACTIVATIONS:
        result = train_dl_activation(X_train, y_train, X_test, y_test, TARGET_COLUMNS, activation)
        dl_results[activation] = result
        for target, m in result["metrics"].items():
            print(f"  [mlp_{activation}] {target}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} R2={m['r2']:.3f}")

    best_activation = min(
        dl_results,
        key=lambda a: np.mean([dl_results[a]["metrics"][t]["rmse"] for t in TARGET_COLUMNS]),
    )
    print(f"\nMejor activacion (menor RMSE promedio): {best_activation.upper()}")

    ensemble_metrics = _load_ensemble_metrics()

    comparison_rows: list[dict] = []
    for target, m in baseline_result["metrics"].items():
        comparison_rows.append({"approach": "ridge_baseline", "target": target, **m})
    for activation, result in dl_results.items():
        for target, m in result["metrics"].items():
            comparison_rows.append({"approach": f"mlp_{activation}", "target": target, **m})
    if ensemble_metrics is not None:
        for target, m in ensemble_metrics.items():
            comparison_rows.append({"approach": "xgboost_catboost_ensemble", "target": target, **m})
    else:
        print("\n[aviso] outputs/reports/model_metrics.json no encontrado -- omitiendo fila del ensamble "
              "(correr `python -m src.modeling` primero para incluirla en la comparacion).")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "dl_baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2, ensure_ascii=False)

    db_path = persist_comparison_duckdb(comparison_rows)
    print(f"\nComparacion persistida en DuckDB: {db_path}")
    print(f"Comparacion persistida en JSON: {REPORTS_DIR / 'dl_baseline_comparison.json'}")

    loss_histories = {a: r["loss_history"] for a, r in dl_results.items()}
    loss_plot_path = save_loss_curves_plot(loss_histories)
    loss_gif_path = save_loss_curves_animation(loss_histories)
    pred_plot_path = save_predicted_vs_actual_plot(
        y_test, dl_results[best_activation]["y_pred"], TARGET_COLUMNS, best_activation
    )
    print(f"Grafico de curvas de loss: {loss_plot_path}")
    print(f"Animacion de curvas de loss: {loss_gif_path}")
    print(f"Grafico predicho-vs-real: {pred_plot_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dl_results[best_activation]["model"].state_dict(), MODELS_DIR / f"mlp_{best_activation}.pt")
    print(f"Modelo MLP guardado en: {MODELS_DIR / f'mlp_{best_activation}.pt'}")


if __name__ == "__main__":
    main()
