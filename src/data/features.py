from __future__ import annotations
import pandas as pd, numpy as np
from ..config import load_config
from ..utils.io import ensure_dir, save_table
from .ingest import load_sources

def _ensure_cols(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    for c, dtype in spec.items():
        if c not in df.columns:
            df[c] = pd.Series([np.nan] * len(df), dtype=dtype if "Int" not in dtype else "float64")
    return df

def _add_season_week_from_game_id(df: pd.DataFrame, game_id_col: str = "game_id") -> pd.DataFrame:
    """
    Si el DF tiene game_id con formato YYYY_WW_AWAY_HOME, deriva season/week si faltan.
    """
    if game_id_col in df.columns and (("season" not in df.columns) or ("week" not in df.columns)):
        parts = df[game_id_col].astype(str).str.split("_", expand=True)
        # Evita crashear si no cumple el formato
        if parts.shape[1] >= 2:
            try:
                df["season"] = df.get("season", parts[0].astype(int))
            except Exception:
                df["season"] = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
            try:
                df["week"] = df.get("week", parts[1].astype(int))
            except Exception:
                df["week"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    return df

def build_player_game_dataset(output_path: str | None = None, prop_type: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    prop = prop_type or cfg.get("target.prop_type", "receiving")

    src = load_sources()

    # --- Base de box scores (jugador–partido)
    bs = src["box_scores"].copy()
    bs = _ensure_cols(bs, {
        "game_id": "object","season":"Int64","week":"Int64","team":"object","opp":"object",
        "player_id":"object","player":"object","position":"object",
        "routes":"float64","targets":"float64","target_share":"float64","air_yards":"float64",
        "rush_att":"float64","snap_pct":"float64","outcome_yards":"float64",
        "receiving_yards":"float64","rushing_yards":"float64",
        "opp_vs_pass_epa":"float64","opp_vs_run_epa":"float64",
    })
    # En caso de que season/week no vengan, intenta derivarlos del game_id
    bs = _add_season_week_from_game_id(bs, "game_id")

    # --- Vegas (nivel juego/equipo) -> derivar season/week si no existen y unir por (season, week, team)
    vegas = src.get("vegas_lines", pd.DataFrame()).copy()
    if not vegas.empty:
        vegas = _add_season_week_from_game_id(vegas, "game_id")
        # Asegura columnas necesarias
        for c in ["season","week","team","vegas_total","vegas_spread"]:
            if c not in vegas.columns:
                if c in ("vegas_total","vegas_spread"):
                    vegas[c] = np.nan
                else:
                    vegas[c] = pd.NA
        vegas = vegas[["season","week","team","vegas_total","vegas_spread"]].drop_duplicates()
        # Merge por (season, week, team)
        bs = bs.merge(vegas, on=["season","week","team"], how="left")
    else:
        bs["vegas_total"] = np.nan
        bs["vegas_spread"] = np.nan

    # --- Weather (nivel juego) -> derivar season/week y unir por (season, week)
    weather = src.get("weather", pd.DataFrame()).copy()
    if not weather.empty:
        weather = _add_season_week_from_game_id(weather, "game_id")
        for c in ["season","week","weather_temp","weather_wind"]:
            if c not in weather.columns:
                weather[c] = np.nan
        weather = weather[["season","week","weather_temp","weather_wind"]].drop_duplicates()
        bs = bs.merge(weather, on=["season","week"], how="left")
    else:
        bs["weather_temp"] = np.nan
        bs["weather_wind"] = np.nan

    # --- Features de ritmo/tendencia (si no existen)
    if "team_pass_rate_neutral" not in bs.columns:
        bs["team_pass_rate_neutral"] = np.nan
    if "team_pace" not in bs.columns:
        bs["team_pace"] = np.nan

    # --- Etiqueta según tipo de prop
    if prop == "receiving" and "receiving_yards" in bs.columns:
        bs["outcome_yards"] = bs["outcome_yards"].fillna(bs["receiving_yards"])
    elif prop == "rushing" and "rushing_yards" in bs.columns:
        bs["outcome_yards"] = bs["outcome_yards"].fillna(bs["rushing_yards"])

    # --- Selección final según schema de config
    keep = cfg.get("schema.columns")
    df = bs[keep].copy()

    if output_path:
        save_table(df, output_path)
    return df
