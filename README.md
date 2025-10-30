# ml-nfl-picks

Pipeline + modelo (LightGBM) y dashboard en Streamlit para generar **picks de props NFL** a partir de datos históricos (2018–2024), con **líneas demo** y evaluación rápida de precisión.

> **Estado 2025:** `nfl_data_py` aún no publica `weekly 2025`; el repo está listo para integrar 2025 en cuanto aparezca. Incluye plan de **modo pregame** (rolling histórico + vegas) y **fallback**.

---

## ✨ Qué hace

* **Ingesta** de datos (Kaggle + `nfl_data_py`) y **parcheo** de vegas/clima.
* **Feature engineering**: une box scores ↔ vegas ↔ weather + métricas de equipo/rival.
* **Modelo** LightGBM que predice **yardas** por jugador.
* **Picks** con probabilidad de **Over** y **edge** (odds demo −110).
* **Dashboard Streamlit** para filtrar por temporada/semana/posición, exportar CSV y ver KPIs: **MAE**, **%±10y**, **hit rate** (demo), **Prob media** y **Edge medio**.

---

## 🗂 Estructura

```
src/
  data/
    ingest.py        # carga de tablas locales (csv/parquet)
    features.py      # dataset jugador-juego (merge vegas/weather + columnas equipo/rival)
  models/
    train.py         # entrenamiento LightGBM
    predict.py       # inferencia (genera pred_yards)
scripts/
  make_dataset.py    # ingesta desde Kaggle + nfl_data_py, escribe raw/external
dashboard/
  streamlit_app.py   # dashboard de picks (carga CSV/Parquet, KPIs, descarga)
data/                # (gitignored) external/raw/processed locales
models/              # (gitignored) artifacts de modelo entrenado
```

---

## 🔧 Requisitos & entorno

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows PowerShell

# Paquetes clave fijados (compatibles con nfl-data-py 0.3.3)
pip install -U pip
pip install -r requirements.txt
```

**Recomendado en `requirements.txt` (además de lo que exporte tu venv):**

```
pandas==1.5.3
pyarrow==21.0.0
nfl-data-py==0.3.3
lightgbm==4.5.0
scikit-learn==1.5.2
streamlit==1.51.0
numpy==1.26.4
scipy==1.13.1
joblib==1.4.2
```

> `nfl-data-py 0.3.3` requiere `pandas < 2.0`, por eso se fija `1.5.3`.

---

## 🚀 Uso rápido (2018–2024)

### 1) Construir datasets desde Kaggle + nfl_data_py

* Descarga **spreadspoke_scores** de Kaggle y colócalo en:

  * `data/external/spreadspoke_scores.csv`
* Coloca tu mapeo de equipos:

  * `data/external/NFL Teams.csv`

Luego:

```bash
python scripts/make_dataset.py \
  --kaggle-csv "data/external/spreadspoke_scores.csv" \
  --teams-csv  "data/external/NFL Teams.csv" \
  --years 2018 2019 2020 2021 2022 2023 2024
```

Esto escribe:

* `data/external/vegas_lines.csv` y `data/external/weather.csv`
* `data/raw/box_scores.csv`

### 2) Dataset procesado (player-game)

```bash
python - <<'PY'
from src.data.features import build_player_game_dataset
build_player_game_dataset("data/processed/player_games.parquet", prop_type="receiving")
print("OK processed")
PY
```

### 3) Entrenar modelo

```bash
python - <<'PY'
from src.models.train import train_model
mae = train_model("data/processed/player_games.parquet", "models/artifacts")
print("MAE CV:", mae)
PY
```

### 4) Generar picks (líneas demo por posición)

```bash
python - <<'PY'
import pandas as pd, numpy as np
from scipy.stats import norm
from src.models.predict import predict

pred = predict("data/processed/player_games.parquet", "models/artifacts/model.joblib")
pred = pred.query("season == 2024").copy()

# Líneas DEMO (ajusta si tienes lines reales)
LINES = {"WR": 60.5, "TE": 35.5, "RB": 55.5, "QB": 255.5}
SIGMA  = {"WR": 15.0, "TE": 12.0, "RB": 14.0, "QB": 35.0}

pred["line"]  = pred["position"].map(LINES).fillna(50.5)
pred["sigma"] = pred["position"].map(SIGMA).fillna(15.0)
pred["prob_over"] = 1 - norm.cdf(pred["line"], loc=pred["pred_yards"], scale=pred["sigma"])

def american_to_payout(odds): return (100/(-odds)) if odds<0 else (odds/100)
payout = american_to_payout(-110)
pred["edge"] = pred["prob_over"]*payout - (1 - pred["prob_over"])

# Filtro suave de calidad (evita WR/TE sin uso en ruta)
pred = pred[(pred["position"].isin(["RB"])) |
            (~pred["position"].isin(["RB"]) & (pred["targets"].fillna(0) >= 2))]

picks = pred.query("prob_over >= 0.58 and edge >= 0.08")[
    ["season","week","player","team","opp","position",
     "line","pred_yards","prob_over","edge","outcome_yards"]
].sort_values(["season","week","edge"], ascending=[True,True,False])

picks.to_csv("data/processed/picks_2024.csv", index=False)
print("OK -> data/processed/picks_2024.csv", len(picks))
PY
```

### 5) Dashboard (Streamlit)

```bash
streamlit run dashboard/streamlit_app.py
```

* Carga `data/processed/picks_2024.csv` (o el más reciente).
* Filtra por **Season / Week / Position** y ajusta **Prob ≥** y **Edge ≥**.
* Exporta el CSV filtrado desde el botón de descarga.

---

## 📏 Demo Accuracy (importante)

En esta fase, el dashboard muestra una **Demo Over/Under Accuracy** que compara `pred_yards` contra **líneas fijas de demostración por posición** (p. ej., WR=60.5, TE=35.5, RB=55.5, QB=255.5).
**No** usa líneas reales del sportsbook; sirve solo como *sanity check* interno del modelo.

* **Calidad del modelo (regresión):** MAE (|outcome − pred|), % dentro ±X yardas.
* **Éxito del pick:** se evalúa contra **la línea** (`outcome_yards ≥ line` para overs).

Si generaste evaluación:

```bash
python - <<'PY'
import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss

SEASON = 2024
picks = pd.read_csv(f"data/processed/picks_{SEASON}.csv")
games = pd.read_parquet("data/processed/player_games.parquet")

eval_df = picks.merge(
    games[["season","week","player","team","outcome_yards"]],
    on=["season","week","player","team"], how="left"
).dropna(subset=["outcome_yards"])

mae = mean_absolute_error(eval_df["outcome_yards"], eval_df["pred_yards"])
eval_df["pred_over"] = eval_df["pred_yards"] > eval_df["line"]
eval_df["real_over"] = eval_df["outcome_yards"] > eval_df["line"]
acc = (eval_df["pred_over"] == eval_df["real_over"]).mean()

try:
    brier = brier_score_loss(eval_df["real_over"].astype(int), eval_df["prob_over"].clip(1e-6,1-1e-6))
except Exception:
    brier = float("nan")

eval_df.to_csv(f"data/processed/eval_rows_{SEASON}.csv", index=False)
print(f"Rows: {len(eval_df)} | MAE: {mae:.2f} | Demo hit: {acc:.3f} | Brier: {brier:.3f}")
PY
```

---

## 🧪 Troubleshooting

* **`nfl_data_py` + Pandas 2.x** → fija `pandas==1.5.3`.
* **No ves `outcome_yards` en el dashboard** → tu `picks_*.csv` no lo trae.
  Solución rápida: merge automático en la app o reescribir `picks_*.csv`:

  ```bash
  python - <<'PY'
  import pandas as pd
  p = pd.read_csv("data/processed/picks_2024.csv")
  g = pd.read_parquet("data/processed/player_games.parquet")[["season","week","player","team","outcome_yards"]]
  p2 = p.merge(g, on=["season","week","player","team"], how="left")
  p2.to_csv("data/processed/picks_2024.csv", index=False)
  print("picks_2024.csv actualizado con outcome_yards")
  PY
  ```
* **Streamlit no recarga** → usa el botón **Rerun** o limpia cache (**☰ → Clear cache**).

---

## 📅 Plan 2025 y “modo pregame”

* **Cuando liberen weekly 2025:** volver a correr `scripts/make_dataset.py` incluyendo 2025.
* **Modo pregame (futuro):**

  * Features **rolling** (4–8 semanas) + vegas (totals/spread/team rates).
  * Ingesta de **player_lines.csv** reales por semana (merge por season/week/team/player).
  * Picks solo para **juegos futuros** (sin `outcome_yards`).
  * Evaluación postgame automática (comparar contra línea real).

---

## ✅ Roadmap corto

**Fase 1.1**

* Congelar `requirements.txt` y primer commit (pipeline + dashboard 2018–2024).
* Subir `data/sample/picks_2024_sample.csv` para demo.

**Fase 1.2**

* Mejoras de UX: botón “Reset filtros”, badges por posición, orden clicable, nombre de descarga con `w{week}_p{prob}_e{edge}`.

**Fase 2**

* Ingesta de **líneas reales** (`player_lines.csv`) y recalcular `prob_over`/`edge`.
* **Modo pregame** + ocultar outcome hasta jugarse el partido.

**Fase 3**

* Integrar 2025 cuando esté disponible.
* Backtest semanal con curva de **cumulative edge** y **hit rate** por umbral.


