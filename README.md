# ML NFL — Sprint 0

Proyecto base para modelar **player props** (yardas por recepción/carrera) con pipeline semanal.

> **Fecha:** 2025-10-30

## Estructura

```
ml-nfl-sprint0/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .env.example
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ raw/
│  ├─ external/
│  ├─ interim/
│  └─ processed/
├─ src/
│  ├─ config.py
│  ├─ utils/
│  │  ├─ io.py
│  │  └─ scoring.py
│  ├─ data/
│  │  ├─ ingest.py
│  │  └─ features.py
│  └─ models/
│     ├─ train.py
│     ├─ evaluate.py
│     └─ predict.py
├─ notebooks/
│  ├─ 00_setup.ipynb
│  ├─ 10_build_dataset.ipynb
│  ├─ 20_train_model.ipynb
│  └─ 30_generate_picks.ipynb
├─ dashboard/
│  └─ streamlit_app.py
├─ docs/
│  ├─ schema.md
│  └─ dashboard_spec.md
└─ tests/
   └─ test_dummy.py
```

## Pasos rápidos
1. Crear entorno e instalar deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```
2. Rellenar fuentes en `configs/default.yaml` (rutas de CSV/Parquet).
3. Ejecutar cuaderno `notebooks/10_build_dataset.ipynb`.
4. Entrenar con `notebooks/20_train_model.ipynb`.
5. Generar picks con `notebooks/30_generate_picks.ipynb`.
6. Dashboard: `streamlit run dashboard/streamlit_app.py`.
