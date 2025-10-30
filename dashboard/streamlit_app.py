# dashboard/streamlit_app.py
import os
import glob
import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="NFL Picks Dashboard (v1)",
    page_icon="🏈",
    layout="wide",
)

# -----------------------------
# Utilidades
# -----------------------------
NUM_COLS = ["line", "pred_yards", "outcome_yards", "prob_over", "edge"]
SAFE_DISPLAY_COLS = [
    "season", "week", "player", "team", "opp", "position",
    "line", "pred_yards", "outcome_yards",
    "err", "abs_err", "prob_over", "edge",
    "pred_over_flag", "real_over", "pick_hit"
]

def read_any(path_or_buffer):
    """Lee CSV o Parquet según extensión o tipo de buffer."""
    if isinstance(path_or_buffer, (str, os.PathLike)):
        p = str(path_or_buffer)
        if p.lower().endswith(".parquet"):
            return pd.read_parquet(p)
        return pd.read_csv(p)
    # Subido por uploader: inspecciona nombre si lo tiene
    name = getattr(path_or_buffer, "name", "")
    if name.lower().endswith(".parquet"):
        return pd.read_parquet(path_or_buffer)
    return pd.read_csv(path_or_buffer)

def find_default_picks():
    """Busca un archivo de picks en data/processed (prioriza picks_YYYY.csv)."""
    candidates = sorted(glob.glob("data/processed/picks_*.csv"))
    if candidates:
        return candidates[-1]
    # fallback genérico
    if os.path.exists("data/processed/picks.csv"):
        return "data/processed/picks.csv"
    return None

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_derived_columns(df):
    """Agrega err, abs_err, real_over, pred_over_flag, pick_hit si hay columnas base."""
    need = {"line", "pred_yards", "outcome_yards"}
    if need.issubset(df.columns):
        df["err"] = df["outcome_yards"] - df["pred_yards"]
        df["abs_err"] = df["err"].abs()
        df["real_over"] = (df["outcome_yards"] > df["line"]).astype(int)
        df["pred_over_flag"] = (df["pred_yards"] > df["line"]).astype(int)
        df["pick_hit"] = (df["real_over"] == df["pred_over_flag"]).astype(int)
    return df

def dedup_picks(df):
    """Quita duplicados típicos, mantiene el de mayor edge/prob si existen."""
    sort_key = []
    # Orden fuerte por edge y prob_over si existen
    if "edge" in df.columns:
        sort_key.append(df["edge"].fillna(-1))
    if "prob_over" in df.columns:
        sort_key.append(df["prob_over"].fillna(-1))
    if not sort_key:
        return df
    df = df.iloc[np.lexsort(tuple(sort_key))]  # orden estable asc por cada clave
    # Mantener el último (el de mayor edge/prob) por llave básica
    dedup_keys = [c for c in ["season", "week", "player", "team"] if c in df.columns]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
    return df

# -----------------------------
# Barra lateral (controles)
# -----------------------------
st.sidebar.header("⚙️ Controles")

default_path = find_default_picks()
file_opt_label = default_path if default_path else "Sube un archivo"
uploaded = st.sidebar.file_uploader(
    "Archivo de picks",
    type=["csv", "parquet"],
    label_visibility="visible"
)

path_select = None
if uploaded is None:
    # Permite elegir un archivo local conocido (si se encontró)
    if default_path:
        path_select = st.sidebar.selectbox(
            "Archivo activo",
            options=[default_path],
            index=0
        )
else:
    path_select = uploaded

if path_select is None:
    st.info("Sube un CSV/Parquet con columnas: player, team, opp, position, line, pred_yards, prob_over, edge (y ojalá outcome_yards).")
    st.stop()

# -----------------------------
# Carga y preparación
# -----------------------------
try:
    raw_df = read_any(path_select)
except Exception as e:
    st.error(f"Error leyendo el archivo: {e}")
    st.stop()

df = raw_df.copy()
df = coerce_numeric(df, NUM_COLS)
df = add_derived_columns(df)
df = dedup_picks(df)

# Asegura tipos mínimos
for c in ["season", "week"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")

# -----------------------------
# Encabezado
# -----------------------------
st.title("🏈 NFL Picks Dashboard (v1)")
active_name = getattr(path_select, "name", path_select)
st.caption(f"Archivo activo: `{active_name}`")

# -----------------------------
# Filtros principales
# -----------------------------
# Season
if "season" in df.columns and df["season"].notna().any():
    seasons = sorted(df["season"].dropna().unique().tolist())
    season_sel = st.sidebar.selectbox("Season", seasons, index=len(seasons)-1)
    df = df[df["season"] == season_sel]
else:
    season_sel = None

# Week
if "week" in df.columns and df["week"].notna().any():
    weeks = sorted(df["week"].dropna().unique().tolist())
    week_sel = st.sidebar.selectbox("Week", weeks)
    df = df[df["week"] == week_sel]

# Position
if "position" in df.columns and df["position"].notna().any():
    pos_opts = sorted(df["position"].dropna().unique().tolist())
    pos_sel = st.sidebar.multiselect("Position", options=pos_opts, default=pos_opts)
    if pos_sel:
        df = df[df["position"].isin(pos_sel)]

# Sliders de prob y edge
# Nota: si prob_over/edge no existen, sliders no filtran
p_min = st.sidebar.slider("Prob ≥ (demo)", 0.00, 1.00, 0.58, 0.01)
e_min = st.sidebar.slider("Edge ≥ (demo)", -1.00, 1.00, 0.08, 0.01)

if "prob_over" in df.columns:
    df = df[df["prob_over"].fillna(0) >= p_min]
if "edge" in df.columns:
    df = df[df["edge"].fillna(-1) >= e_min]

view = df.copy()

# -----------------------------
# KPIs
# -----------------------------
def safe_mean(series):
    return float(series.mean()) if len(series) else float("nan")

# Métricas de modelado
if {"abs_err", "pick_hit"}.issubset(view.columns):
    mae = safe_mean(view["abs_err"])
    within10 = float((view["abs_err"] <= 10).mean()) if len(view) else float("nan")
    pick_acc = safe_mean(view["pick_hit"])
else:
    mae = within10 = pick_acc = float("nan")

# Métricas de probabilidades
prob_mean = safe_mean(view["prob_over"]) if "prob_over" in view.columns else float("nan")
edge_mean = safe_mean(view["edge"]) if "edge" in view.columns else float("nan")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Filas visibles", len(view))
c2.metric("MAE (yds)", f"{mae:.2f}" if np.isfinite(mae) else "—")
c3.metric("% dentro ±10y", f"{within10*100:.1f}%" if np.isfinite(within10) else "—")
c4.metric("Pick hit rate", f"{pick_acc*100:.1f}%" if np.isfinite(pick_acc) else "—")
c5.metric("Prob. media (demo)", f"{prob_mean*100:.1f}%" if np.isfinite(prob_mean) else "—")
c6.metric("Edge medio (demo)", f"{edge_mean:.3f}" if np.isfinite(edge_mean) else "—")

st.markdown(
    "_Nota: **Pick hit rate** compara la dirección del modelo vs la **línea** (over/under). "
    "**MAE** y **±10y** miden la calidad de la predicción en yardas._"
)

# -----------------------------
# Tabla
# -----------------------------
display_cols = [c for c in SAFE_DISPLAY_COLS if c in view.columns]
if not display_cols:
    display_cols = list(view.columns)

# Orden por edge desc si existe
if "edge" in view.columns:
    view = view.sort_values(["edge", "prob_over"], ascending=[False, False])

st.dataframe(
    view[display_cols],
    use_container_width=True,
    height=580
)

# -----------------------------
# Descarga CSV filtrado
# -----------------------------
fname_bits = [os.path.splitext(os.path.basename(active_name))[0]]
if season_sel is not None:
    fname_bits.append(f"s{season_sel}")
if "week" in df.columns and len(df["week"].unique()) == 1:
    fname_bits.append(f"w{int(df['week'].iloc[0])}")
fname_bits.append(f"p{int(round(p_min*100)):02d}")
fname_bits.append(f"e{int(round(e_min*100)):02d}")
dl_name = "_".join(fname_bits) + ".csv"

csv_bytes = view.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar CSV (filtrado)",
    data=csv_bytes,
    file_name=dl_name,
    mime="text/csv",
)
