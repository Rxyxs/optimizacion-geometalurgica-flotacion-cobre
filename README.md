# optimizacion-geometalurgica-flotacion-cobre

Pipeline monolítico ("Single-Phase Master Pipeline") de optimización
geometalúrgica para flotación de **Cobre (Cu)** y **Molibdeno (Mo)**:
simula un block model + telemetría de celda de flotación, limpia y
enriquece los datos, entrena un ensamble multi-output de Recuperación,
ejecuta un motor de optimización prescriptiva (Algoritmo Genético) para
bloques de baja recuperación, y genera explicabilidad SHAP + un reporte
bilingüe (ES/EN) — todo con **un solo comando**, sin intervención humana.

## 🎯 Problema de negocio

Predecir la Recuperación metalúrgica de Cu y Mo a partir de la geología del
bloque y las condiciones operativas de la celda de flotación (pH,
reactivos, tamaño de partícula P80, aire, % sólidos), e identificar —para
los bloques con recuperación de Cu predicha por debajo del umbral de
negocio (82%)— el ajuste operativo que maximiza la recuperación sin exceder
el presupuesto de insumos.

## 🏗️ Arquitectura del pipeline

```
1. data_generator.py       Block model + telemetría sintética (50k bloques)
        │
        ▼
2. wrangling.py             Filtro de Kalman (sensores) + imputación
        │                   espacial (KDTree) + Isolation Forest (outliers)
        ▼
3. feature_engineering.py   Ratios de mineralogía + interacciones SGI×
        │                   reactivos + perturbaciones de aire (Wavelet)
        ▼
4. modeling.py               XGBoost + CatBoost multi-output (Cu%, Mo%),
        │                    validados con TimeSeriesSplit (walk-forward)
        ▼
5. optimizer.py              Algoritmo Genético (DEAP): objetivo único (Cu)
        │                    + frente de Pareto NSGA-II (Cu vs Mo)
        ▼
6. explainability.py         SHAP global + drivers de pérdida de metal +
                             dashboard PDF/PNG + reporte bilingüe (ES/EN)

        (opcional, no es parte del batch)
        api.py                Servicio FastAPI: scoring y optimización
                               (ambos motores) en tiempo real, bajo demanda
```

`master_pipeline.py` ejecuta las 6 etapas del batch en secuencia. Cada
módulo también corre de forma independiente (`python -m src.<modulo>`)
para depuración, leyendo/escribiendo sus artefactos en `data/` y
`outputs/`. `api.py` es un servicio aparte (`uvicorn`, no un `main()` de
batch) que reutiliza los mismos artefactos y motores de optimización.

## 🛠️ Stack

- Python 3.11+
- Polars + PyArrow — ingesta y feature engineering de alta velocidad
- SciPy (`cKDTree`, `lfilter`) + PyWavelets — filtros de Kalman, procesos
  AR(1), imputación espacial e indicadores de perturbación por Wavelet
- scikit-learn — Isolation Forest, `TimeSeriesSplit`, métricas
- XGBoost + CatBoost — modelado multi-output (Cu% y Mo% simultáneos)
- DEAP — Algoritmo Genético: objetivo único (`selTournament`) y
  multi-objetivo NSGA-II (`selNSGA2` + `selTournamentDCD`)
- SHAP — explicabilidad técnica global y local
- Matplotlib + Seaborn — dashboard exportado a PDF/PNG
- FastAPI + httpx — servicio de scoring/optimización en tiempo real

## 📁 Estructura

```
optimizacion-geometalurgica-flotacion-cobre/
├── data/
│   ├── raw/                          # block model + telemetria cruda (generado)
│   └── processed/                    # datos limpios + features (generado)
├── outputs/
│   ├── models/                       # ensamble entrenado + dashboard PDF/PNG
│   ├── reports/                      # metricas, SHAP, recomendaciones, reporte bilingue
│   └── plots/                        # dashboard.png / .pdf
├── src/
│   ├── data_generator.py             # block model + telemetria (Cu/Mo)
│   ├── wrangling.py                  # Kalman, imputacion espacial, Isolation Forest
│   ├── feature_engineering.py        # mineralogia, interacciones, Wavelet
│   ├── modeling.py                   # XGBoost + CatBoost multi-output, walk-forward CV
│   ├── optimizer.py                  # Algoritmo Genetico: objetivo unico + Pareto (NSGA-II)
│   ├── explainability.py             # SHAP, dashboard, reporte bilingue
│   ├── master_pipeline.py            # orquestador unico (batch, 6 etapas)
│   └── api.py                        # servicio FastAPI: scoring/optimizacion en tiempo real
├── requirements.txt
└── README.md
```

## 🚀 Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota:** `xgboost` está fijado a `<3.0.0`. La serie 3.x cambió el
> formato interno de `base_score` en el volcado JSON del modelo de forma
> incompatible con `shap.TreeExplainer` en la version de `shap` usada aqui
> (error `could not convert string to float: '[8.78...E1]'`). Validado con
> `xgboost==2.1.4`.

## ▶️ Uso

### Pipeline completo (un solo comando)

```powershell
python -m src.master_pipeline
```

Corre las 6 etapas de punta a punta (~65 segundos en una corrida típica) y
deja todos los artefactos listos en `data/` y `outputs/`. Un error en
cualquier etapa detiene el pipeline con el traceback completo — no hay
reintentos silenciosos ni continuación con datos parciales.

### Etapas individuales (para depuración)

```powershell
python -m src.data_generator        # data/raw/block_model_flotation_raw.parquet
python -m src.wrangling             # data/processed/block_model_flotation_clean.parquet
python -m src.feature_engineering   # data/processed/block_model_flotation_features.parquet
python -m src.modeling              # outputs/models/geometallurgical_ensemble.joblib
python -m src.optimizer             # outputs/reports/{optimization,pareto}_recommendations.{parquet,csv}
python -m src.explainability        # outputs/reports/*.json,*.txt + outputs/plots/dashboard.*
```

### Servicio en tiempo real (FastAPI)

```powershell
uvicorn src.api:app --reload
```

Requiere haber corrido el pipeline (o al menos hasta `modeling.py`) al
menos una vez, para que existan `outputs/models/geometallurgical_ensemble
.joblib` y `data/processed/block_model_flotation_features.parquet`.

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y n° de bloques cargados |
| `GET /blocks/at-risk?limit=50` | Bloques con Cu predicho bajo el umbral, peor a mejor |
| `GET /blocks/{block_id}/score` | Recuperación Cu/Mo predicha + nivel de riesgo |
| `GET /blocks/{block_id}/optimize` | GA objetivo único: mejor (pH, reactivos, P80) para maximizar Cu |
| `GET /blocks/{block_id}/optimize/pareto` | NSGA-II: frente de Pareto completo Cu-vs-Mo |

```powershell
curl http://127.0.0.1:8000/blocks/BLK-041590/optimize/pareto
```

## 🗄️ Datos simulados

**Block model** (50.000 bloques, coordenadas x/y/banco): `cu_grade_pct`,
`mo_grade_pct`, `sgi_kwh_t` (dureza, proxy Bond Work Index),
`pyrite_pct`, `chalcopyrite_frac`/`bornite_frac` (mineralogía de sulfuros
de cobre) — con continuidad espacial real (`scipy.spatial.cKDTree`
promediando cada bloque con sus 8 vecinos más cercanos).

**Telemetría de celda**: `ph`, `air_flow_m3_h`, `pct_solids`, `p80_um` como
procesos AR(1) de media-reversión (`scipy.signal.lfilter`) — representan
el valor **verdadero** de planta, sobre el que se agrega ruido de
instrumento. La recuperación responde al valor verdadero, no al ruidoso,
de modo que el filtro de Kalman tiene un efecto medible y honesto. El
flujo de aire además incluye ~0.4% de bloques con eventos de perturbación
real (paradas de soplador, bloqueos), para que la detección por Wavelet en
`feature_engineering.py` tenga señal genuina que encontrar.

**Reactivos**: `collector_g_t` (correlacionado con el contenido de
sulfuros) y `frother_g_t`.

**Recuperación** (`cu_recovery_pct`, `mo_recovery_pct`): combinación de
funciones de respuesta no lineales (isoterma de saturación para reactivos,
curvas con óptimo para P80/aire/pH/sólidos, penalización de selectividad
por piritas) pasada por una **transformación logística** que acota el
resultado a un rango realista sin necesidad de recortar (clip) la cola —
sumar directamente varios términos acotados en [0,1] con pesos grandes
genera colas irrealmente extremas (recuperaciones &gt;100%); la sigmoide
comprime eso de forma suave.

## ⚙️ Motor de optimización prescriptiva

Para los bloques con Cu predicho bajo el umbral de negocio (82%), un
Algoritmo Genético (DEAP) busca `pH`, `collector_g_t`, `frother_g_t` y
`p80_um` óptimos, sujeto a un presupuesto de reactivos
(`REAGENT_BUDGET_USD_PER_T = 0.22`, con precios de referencia
`COLLECTOR_PRICE_USD_PER_KG = 2.6` y `FROTHER_PRICE_USD_PER_KG = 3.9`).

Por rendimiento, la evaluación de fitness de **toda la población de una
generación se hace en un solo lote** (`model.predict()` vectorizado) en
vez de individuo por individuo — casi tan rápido como predecir una fila
para un ensamble de árboles. El motor se acota a los `MAX_BLOCKS_TO_OPTIMIZE
= 150` bloques de peor recuperación (no los miles que puedan calificar),
para mantener el tiempo de ejecución del pipeline predecible; es una
decisión de diseño explícita, no un límite oculto.

**Modo multi-objetivo (Pareto Cu-vs-Mo):** `run_pareto_genetic_algorithm`
usa NSGA-II (`tools.selNSGA2` + `tools.selTournamentDCD`) con
`creator.FitnessMulti(weights=(1.0, 1.0))` para maximizar Cu **y** Mo
simultáneamente, devolviendo el frente completo de soluciones no
dominadas en vez de un único óptimo — el operador elige el punto del
trade-off que prefiera. Se corre para los `MAX_BLOCKS_PARETO = 30` peores
bloques (menos que el modo objetivo-único: cada frente es más costoso de
interpretar). En la corrida de referencia, la correlación Cu-vs-Mo
**dentro** de un mismo frente ronda **-0.8**: mejorar Cu implica
sacrificar Mo, porque sus P80 óptimos son distintos (170 µm vs. 140 µm) —
el trade-off es una consecuencia física real del diseño del generador, no
un artefacto numérico.

## 📊 Resultados (corrida de referencia, 50.000 bloques, seed 42)

| Modelo | Target | RMSE | MAE | R² |
|---|---|---|---|---|
| Ensemble (XGBoost+CatBoost) | Cu recovery % | 4.08 | 3.06 | **0.648** |
| Ensemble (XGBoost+CatBoost) | Mo recovery % | 4.65 | 3.71 | **0.669** |

Validado con `TimeSeriesSplit` (5 folds, walk-forward: siempre se entrena
en el pasado y se evalúa en el futuro).

- **9.644 bloques** (19.3%) con Cu predicho bajo el umbral de 82%.
- El motor de optimización procesó los **150 de menor recuperación** y
  recomendó ajustes que elevarían la recuperación de Cu predicha en
  **+27.6 puntos porcentuales en promedio**, sin exceder el presupuesto de
  reactivos (costo promedio recomendado: 0.216 USD/t, bajo el límite de
  0.22 USD/t).
- **Top features SHAP para Cu**: `p80_deviation_from_optimum`, `ph_kf`,
  `air_flow_m3_h_kf`, `pct_solids_kf`, `sgi_pyrite_interaction` — coincide
  con el diseño físico del generador (P80 y pH fuera de óptimo son los
  principales drivers de pérdida de metal).

## 🐞 Nota de depuración (por transparencia)

Durante la calibración se encontró y corrigió un bug real en el generador:
la implementación inicial del proceso AR(1) solo aplicaba el término de
reversión a la media en el primer paso, no en cada paso, causando que las
cuatro variables de sensor decayeran lentamente hacia 0 y quedaran
"pegadas" en el límite inferior de su rango (p. ej. pH promediando ~9.0 en
vez de ~10.6 en todo el dataset). Esto inflaba artificialmente el
porcentaje de bloques bajo el umbral y sesgaba las variables de sensor.
Corregido sumando el término de deriva en cada paso del filtro IIR (ver
`_ar1_process` en `data_generator.py`); los R² reportados arriba son
posteriores a la corrección.

## 🔭 Próximos pasos

- **Reemplazar el generador por telemetría histórica real de planta** —
  pendiente: requiere datos reales que este repositorio no tiene ni puede
  fabricar; sigue siendo el paso crítico antes de cualquier uso productivo.
- ~~Exponer el motor de optimización como servicio (API) para
  recomendaciones en tiempo real~~ — hecho, ver `src/api.py`.
- ~~Extender el Algoritmo Genético a optimización multi-objetivo (Cu y Mo
  simultáneos)~~ — hecho, ver `run_pareto_genetic_algorithm` en
  `src/optimizer.py`.
- Agregar autenticación/rate-limiting a `api.py` antes de cualquier
  despliegue expuesto a red.
- SHAP para el motor de Pareto (hoy la explicabilidad solo cubre el modelo
  de recuperación, no las recomendaciones multi-objetivo).
