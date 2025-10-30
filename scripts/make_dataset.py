#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_dataset.py
Construye datasets intermedios para el proyecto:
  1) vegas_lines.csv y weather.csv desde spreadspoke_scores.csv (Kaggle)
  2) box_scores.csv con weekly player stats:
       - intenta nfl_data_py.import_weekly_data
       - si falla (p.ej. 2025 -> 404), cae a nflverse (player_stats_{year}.parquet)
Uso:
  python scripts/make_dataset.py \
    --kaggle-csv "data/external/spreadspoke_scores.csv" \
    --teams-csv  "data/external/NFL Teams.csv" \
    --years 2018 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# nfl_data_py es opcional; si no existe, el fallback cubrirá
try:
    import nfl_data_py as nfl
except Exception:
    nfl = None


# --------------------------
# Utilidades
# --------------------------
def log(msg: str) -> None:
    print(f"[make_dataset] {msg}")


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# --------------------------
# Paso 1: Vegas & Weather
# --------------------------
def _safe_float(x):
    try:
        if pd.isna(x) or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _build_game_id(season: int | str, week: int | str, home: str, away: str) -> str:
    # game_id: YYYY_WW_AWAY_HOME (como en tu repo)
    s = int(season)
    w = int(week)
    return f"{s:04d}_{w:02d}_{away.upper()}_{home.upper()}"


def build_vegas_and_weather(kaggle_csv: str, teams_info: pd.DataFrame) -> None:
    """
    Lee spreadspoke_scores.csv (Kaggle) y escribe:
      - data/external/vegas_lines.csv  (game_id, team, vegas_total, vegas_spread)
      - data/external/weather.csv      (game_id, weather_temp, weather_wind)
    El CSV de Kaggle cambia a veces nombres; cubrimos variantes comunes.
    """
    if not Path(kaggle_csv).exists():
        raise FileNotFoundError(f"Kaggle CSV not found at {kaggle_csv}")

    log("1) Building vegas_lines & weather from Kaggle...")
    df = pd.read_csv(kaggle_csv)

    # Normalización de columnas comunes en spreadspoke
    # Ver variantes: https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data
    colmap = {
        "schedule_season": "season",
        "schedule_week": "week",
        "team_home": "home_team",
        "team_away": "away_team",
        "team_favorite_id": "fav_team",
        "spread_favorite": "spread_favorite",
        "spread_line": "spread_line",
        "over_under_line": "total",
        "weather_temperature": "weather_temp",
        "weather_wind_mph": "weather_wind",
        "stadium_neutral": "neutral",
        "stadium": "stadium",
    }
    # Renombra solo si existen
    present = {k: v for k, v in colmap.items() if k in df.columns}
    df = df.rename(columns=present)

    # Algunas versiones usan "team" de 3 letras. No forzamos a mapear; usamos tal cual.
    for need in ["season", "week", "home_team", "away_team"]:
        if need not in df.columns:
            raise ValueError(f"Column '{need}' not found in Kaggle CSV")

    # Total y spread
    if "total" not in df.columns:
        df["total"] = np.nan
    if "spread_line" not in df.columns and "spread_favorite" in df.columns:
        # algunas versiones usan spread_favorite como número; úsalo
        df["spread_line"] = df["spread_favorite"]
    elif "spread_line" not in df.columns:
        df["spread_line"] = np.nan
    if "fav_team" not in df.columns:
        df["fav_team"] = np.nan

    # Limpia tipos
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = df["week"].replace("Wildcard", 18).replace("Division", 19).replace("Conference", 20).replace("Superbowl", 21)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["total"] = df["total"].map(_safe_float)
    df["spread_line"] = df["spread_line"].map(_safe_float)

    # game_id
    df["game_id"] = df.apply(
        lambda r: _build_game_id(r["season"], r["week"], r["home_team"], r["away_team"]),
        axis=1,
    )

    # --- vegas_lines por EQUIPO ---
    # Para cada juego, construimos dos filas: home y away
    rows = []
    for _, r in df[["game_id", "season", "week", "home_team", "away_team", "fav_team", "spread_line", "total"]].iterrows():
        gid = r["game_id"]
        total = r["total"]
        fav = str(r["fav_team"]).upper() if pd.notna(r["fav_team"]) else None
        spread = r["spread_line"]

        home = str(r["home_team"]).upper()
        away = str(r["away_team"]).upper()

        # favorite recibe -spread; el otro +spread
        if pd.notna(spread) and fav in {home, away}:
            home_spread = -spread if fav == home else spread
            away_spread = -spread if fav == away else spread
        else:
            # si no hay favorito claro
            home_spread = np.nan if pd.isna(spread) else -spread / 2.0
            away_spread = np.nan if pd.isna(spread) else +spread / 2.0

        rows.append((gid, home, total, home_spread))
        rows.append((gid, away, total, away_spread))

    vegas = pd.DataFrame(rows, columns=["game_id", "team", "vegas_total", "vegas_spread"])

    # --- weather por JUEGO ---
    wcols = []
    if "weather_temp" in df.columns:
        wcols.append("weather_temp")
    if "weather_wind" in df.columns:
        wcols.append("weather_wind")
    if not wcols:
        # crea placeholder
        weather = pd.DataFrame({"game_id": df["game_id"], "weather_temp": np.nan, "weather_wind": np.nan})
    else:
        weather = df[["game_id"] + wcols].copy()
        if "weather_temp" not in weather.columns:
            weather["weather_temp"] = np.nan
        if "weather_wind" not in weather.columns:
            weather["weather_wind"] = np.nan

    # Guardar
    vegas_path = Path("data/external/vegas_lines.csv")
    weather_path = Path("data/external/weather.csv")
    ensure_dir(vegas_path); ensure_dir(weather_path)
    vegas.to_csv(vegas_path, index=False)
    weather.to_csv(weather_path, index=False)
    log(f"Wrote {vegas_path} ({len(vegas)} rows)")
    log(f"Wrote {weather_path} ({len(weather)} rows)")


# --------------------------
# Paso 2: Box scores (weekly)
#       con fallback a nflverse
# --------------------------
def _read_nflverse_weekly(year: int) -> pd.DataFrame:
    """
    Fallback cuando nfl_data_py no tiene el año.
    Prueba rutas nuevas (nflverse-pbp) y viejas (nflfastR-data),
    incluyendo mirrors via raw.githubusercontent.com.
    """
    urls = [
        f"https://github.com/nflverse/nflverse-pbp/raw/master/data/player_stats/player_stats_{year}.parquet",
        f"https://github.com/nflverse/nflfastR-data/raw/master/data/player_stats/player_stats_{year}.parquet",
        f"https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/data/player_stats/player_stats_{year}.parquet",
        f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/player_stats/player_stats_{year}.parquet",
    ]
    last_err = None
    df = None
    for url in urls:
        try:
            df = pd.read_parquet(url, engine="pyarrow")
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err

    # Normalización hacia tu esquema
    rename = {
        "recent_team": "team",
        "opponent_team": "opp",
        "player_id": "player_id",
        "player_name": "player",
        "routes_run": "routes",
        "rushing_attempts": "rush_att",
        # snap share cambia de nombre entre años:
        "offense_snaps_share": "snap_pct",
        "offense_pct": "snap_pct",
    }
    present = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=present)
    return df


def build_box_scores(years: list[int]) -> None:
    if nfl is None:
        raise ImportError("nfl_data_py is not installed. Please `pip install nfl_data_py`.")

    log(f"2) Building box_scores from nfl_data_py (years={years})...")
    frames, ok_years = [], []

    for y in years:
        try:
            # Fuente oficial
            df_y = nfl.import_weekly_data([y])
            print("Downcasting floats.")
        except Exception as e1:
            print(f"[box_scores] nfl_data_py falló para {y}: {e1} -> probando nflverse…")
            try:
                df_y = _read_nflverse_weekly(y)
            except Exception as e2:
                print(f"[box_scores] Omitiendo {y}: {e2}")
                continue

        frames.append(df_y)
        ok_years.append(y)

    if not frames:
        raise RuntimeError("No se pudo importar ningún año de weekly data.")

    stats = pd.concat(frames, ignore_index=True)

    # Renombra a tu esquema/esperado
    rename = {
        "season": "season",
        "week": "week",
        "team": "team",
        "opp": "opp",
        "recent_team": "team",
        "opponent_team": "opp",
        "player_id": "player_id",
        "player": "player",
        "player_name": "player",
        "position": "position",
        "routes": "routes",
        "routes_run": "routes",
        "targets": "targets",
        "target_share": "target_share",
        "air_yards": "air_yards",
        "rush_att": "rush_att",
        "rushing_attempts": "rush_att",
        "snap_pct": "snap_pct",
        "offense_snaps_share": "snap_pct",
        "offense_pct": "snap_pct",
        "receiving_yards": "receiving_yards",
        "rushing_yards": "rushing_yards",
        "game_id": "game_id",
    }
    present = {k: v for k, v in rename.items() if k in stats.columns}
    df = stats.rename(columns=present)

    needed = [
        "game_id","season","week","team","opp","player_id","player","position",
        "routes","targets","target_share","air_yards","rush_att","snap_pct",
        "receiving_yards","rushing_yards"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # game_id sintético si falta
    if "game_id" in df.columns:
        no_gid = df["game_id"].isna()
    else:
        df["game_id"] = np.nan
        no_gid = df["game_id"].isna()

    if no_gid.any():
        df.loc[no_gid, "game_id"] = (
            df.loc[no_gid, "season"].astype("Int64").astype(str).str.zfill(4) + "_" +
            df.loc[no_gid, "week"].astype("Int64").astype(str).str.zfill(2) + "_" +
            df.loc[no_gid, "team"].astype(str).str.upper() + "_" +
            df.loc[no_gid, "opp"].astype(str).str.upper()
        )

    out = Path("data/raw/box_scores.csv")
    ensure_dir(out)
    df = df[needed]
    df.to_csv(out, index=False)
    log(f"[box_scores] Años incluidos: {ok_years}")
    log(f"Wrote {out} ({len(df)} rows)")


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build intermediate datasets (vegas + weather + box scores)")
    p.add_argument("--kaggle-csv", required=True, help="Ruta a spreadspoke_scores.csv")
    p.add_argument("--teams-csv", required=True, help="Ruta a 'NFL Teams.csv' (no obligatorio para este paso, pero lo conservamos)")
    p.add_argument("--years", nargs="+", type=int, required=True, help="Años a importar (e.g., 2018 ... 2025)")
    return p.parse_args()


def main():
    args = parse_args()

    # Cargar teams (por ahora no lo usamos, pero se mantiene por compatibilidad)
    teams_csv = Path(args.teams_csv)
    if teams_csv.exists():
        teams = pd.read_csv(teams_csv)
    else:
        teams = pd.DataFrame()

    build_vegas_and_weather(args.kaggle_csv, teams)

    build_box_scores(args.years)

    log("Done. Now run notebooks/10_build_dataset.ipynb.")


if __name__ == "__main__":
    main()
