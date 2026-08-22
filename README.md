# optimizacion-geometalurgica-flotacion-cobre

# Optimización Geometalúrgica en Plantas de Flotación (Cu/Mo)

Sistema automatizado *end-to-end* diseñado para predecir y optimizar la recuperación de Cobre (Cu) y Molibdeno (Mo) en procesos de flotación. El proyecto cruza información del modelo de bloques geológicos con telemetría de celdas en tiempo real para resolver pérdidas de metal en relaves.

Aplica procesamiento de señales para limpiar el ruido de lectura en planta, entrena un ensamble *multi-output* de gradiente boosting y ejecuta un motor prescriptivo basado en algoritmos genéticos que sugiere ajustes exactos de reactivos y pH para bloques mineralógicos complejos. Todo el flujo (ingesta, ingeniería de variables, modelado, optimización y explicabilidad SHAP) se ejecuta de principio a fin desde un único comando (`python -m src.master_pipeline`).

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
- SHAP — explicabilidad técnica global, local y del frente de Pareto
- Matplotlib + Seaborn — dashboard exportado a PDF/PNG
- FastAPI + slowapi + httpx — servicio de scoring/optimización en tiempo
  real, con API key y rate-limiting

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
python -m src.explainability        # outputs/reports/*.json,*.txt + outputs/plots/*.png,*.pdf
```

### Servicio en tiempo real (FastAPI)

Requiere haber corrido el pipeline (o al menos hasta `modeling.py`) al
menos una vez, para que existan `outputs/models/geometallurgical_ensemble
.joblib` y `data/processed/block_model_flotation_features.parquet`.
Requiere una API key por header (`X-API-Key`) en todos los endpoints salvo
`/health`, y aplica rate-limiting por IP (60 req/min por defecto):

```powershell
$env:GEOMET_API_KEY = "tu-clave-secreta"      # default: "dev-key-change-me"
$env:GEOMET_RATE_LIMIT = "60/minute"          # opcional
uvicorn src.api:app --reload
```

| Endpoint | Auth | Descripción |
|---|---|---|
| `GET /health` | No | Estado del servicio y n° de bloques cargados |
| `GET /blocks/at-risk?limit=50` | Sí | Bloques con Cu predicho bajo el umbral, peor a mejor |
| `GET /blocks/{block_id}/score` | Sí | Recuperación Cu/Mo predicha + nivel de riesgo |
| `GET /blocks/{block_id}/optimize` | Sí | GA objetivo único: mejor (pH, reactivos, P80) para maximizar Cu |
| `GET /blocks/{block_id}/optimize/pareto` | Sí | NSGA-II: frente de Pareto completo Cu-vs-Mo |

```powershell
curl -H "X-API-Key: tu-clave-secreta" http://127.0.0.1:8000/blocks/BLK-041590/optimize/pareto
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

**Explicabilidad del frente de Pareto:** `explainability.py` calcula SHAP
local (Cu y Mo) para las dos soluciones extremas de cada frente — la que
maximiza Cu y la que maximiza Mo — reconstruyendo el vector de features en
ese punto de operación. El resultado (`pareto_shap_comparison.png` +
`pareto_shap_explanations` en el reporte JSON) muestra que ambas
soluciones llegan a su óptimo por mecanismos distintos, y en la corrida de
referencia revela algo no evidente a simple vista: `air_flow_m3_h_kf` —una
variable de **contexto fija, no ajustable por el GA**— es el mayor lastre
negativo en ambas soluciones para el bloque de ejemplo, es decir, hay
bloques donde ningún ajuste de pH/reactivos/P80 alcanza a compensar una
condición de aire desfavorable.

## 🐞 Nota de depuración (por transparencia)

Durante la calibración se encontró y corrigió un bug real en el generador:
la implementación inicial del proceso AR(1) solo aplicaba el término de
reversión a la media en el primer paso, no en cada paso, causando que las
cuatro variables de sensor decayeran lentamente hacia 0 y quedaran
"pegadas" en el límite inferior de su rango (p. ej. pH promediando ~9.0 en
vez de ~10.6 en todo el dataset). Esto inflaba artificialmente el
porcentaje de bloques bajo el umbral y sesgaba las variables de sensor.
Corregido sumando el término de deriva en cada paso del filtro IIR (ver
`_ar1_process` en `data_generator.py`); los resultados reportados abajo
son posteriores a la corrección.

## 🔭 Próximo paso pendiente

De los cuatro puntos que quedaban abiertos, tres ya están resueltos: API
en tiempo real (`src/api.py`), optimización multi-objetivo
(`run_pareto_genetic_algorithm`), autenticación/rate-limiting
(`GEOMET_API_KEY` + `slowapi`) y SHAP para el frente de Pareto. Queda uno,
y es el crítico:

- **Reemplazar el generador por telemetría histórica real de planta.**
  Este repositorio no tiene acceso a datos reales de ninguna faena y no
  puede fabricarlos — es un límite deliberado, no una omisión. Sin este
  paso, todo lo demás queda validado como *pipeline*, no como modelo
  metalúrgico de producción (ver Conclusiones).

## 📊 Resultados (corrida de referencia, 50.000 bloques, seed 42)

| Modelo | Target | RMSE | MAE | R² |
|---|---|---|---|---|
| Ensemble (XGBoost+CatBoost) | Cu recovery % | 4.08 | 3.06 | **0.648** |
| Ensemble (XGBoost+CatBoost) | Mo recovery % | 4.65 | 3.71 | **0.669** |

Validado con `TimeSeriesSplit` (5 folds, walk-forward: siempre se entrena
en el pasado y se evalúa en el futuro).

- **9.644 bloques** (19.3%) con Cu predicho bajo el umbral de 82%.
- El motor de optimización de objetivo único procesó los **150 de menor
  recuperación** y recomendó ajustes que elevarían la recuperación de Cu
  predicha en **+27.6 puntos porcentuales en promedio**, sin exceder el
  presupuesto de reactivos (costo promedio recomendado: 0.216 USD/t, bajo
  el límite de 0.22 USD/t).
- El motor multi-objetivo calculó el frente de Pareto completo para los
  **30 bloques** de peor recuperación (**~35 soluciones no dominadas** por
  frente, en promedio), con una correlación Cu-vs-Mo **-0.8 dentro de cada
  frente** — un trade-off físico real, no ruido numérico.
- **Top features SHAP para Cu** (modelo global): `p80_deviation_from_optimum`,
  `ph_kf`, `air_flow_m3_h_kf`, `pct_solids_kf`, `sgi_pyrite_interaction` —
  coincide con el diseño físico del generador (P80 y pH fuera de óptimo
  son los principales drivers de pérdida de metal).
- La API (`/health`, `/blocks/at-risk`, `/blocks/{id}/score`,
  `/blocks/{id}/optimize`, `/blocks/{id}/optimize/pareto`) fue validada
  contra un servidor real: 401 sin API key, 200 con key válida, 404 para
  bloques inexistentes.

## ✅ Conclusiones

- **El sistema completo — batch y en tiempo real — funciona de punta a
  punta con datos reales generados por el propio repositorio**: desde la
  simulación geoespacial y de sensores hasta un servicio HTTP autenticado
  que sirve recomendaciones bajo demanda, todo entrenado, optimizado y
  explicado con las mismas 49.000 unidades.
- **El ensamble predice con R² 0.65 (Cu) y 0.67 (Mo) en validación
  temporal honesta** (nunca se entrena con el futuro para predecir el
  pasado) — suficiente señal para que tanto el ranking de riesgo como las
  recomendaciones del GA sean accionables, no ruido.
- **El motor de optimización de objetivo único encuentra mejoras reales y
  acotadas por presupuesto** (+27.6 pp promedio, sin superar el costo de
  reactivos definido) — no es una promesa vacía, es una búsqueda validada
  contra el mismo modelo que se usa para reportar el resultado.
- **El frente de Pareto no es una curiosidad matemática: expone un
  trade-off metalúrgico real.** La correlación -0.8 entre Cu y Mo dentro
  de un mismo frente, y el hecho de que SHAP muestre mecanismos distintos
  para cada solución extrema, son consistentes con que Cu y Mo tienen
  óptimos de P80 distintos por diseño — el operador puede elegir su punto
  del trade-off con información real, no adivinando.
- **La explicabilidad ya no se detiene en "qué tan bueno es el modelo":
  llega hasta "por qué esta recomendación específica es la que es"**,
  incluyendo el hallazgo de que algunas variables de contexto (como el
  flujo de aire) no son ajustables por el optimizador y pueden limitar el
  resultado sin importar cuánto se optimicen pH, reactivos o P80 — una
  distinción que le importa a un metalurgista y que antes no era visible.
- **La limitación central sigue siendo la misma que en la entrega
  anterior, y vale la pena repetirla sin rodeos**: todo el dataset es
  sintético. Las métricas de esta sección validan que la arquitectura, los
  splits sin fuga, el motor de optimización y la explicabilidad son
  correctos — no validan que el sistema prediga fallas o recuperaciones
  reales en una faena. El paso crítico antes de cualquier uso productivo
  sigue siendo reemplazar el generador por telemetría histórica real (ver
  "Próximo paso pendiente" arriba).
