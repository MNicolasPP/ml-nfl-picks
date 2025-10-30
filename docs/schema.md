# Esquema de Dataset (Jugador–Partido)

- **game_id**: str — Identificador único de juego.
- **season, week**: int — Temporada y semana.
- **team, opp**: str — Equipo y rival (abbr).
- **player_id, player, position**: str — ID, nombre y posición (WR/RB/TE).
- **routes, targets, target_share, air_yards**: float — Uso aéreo.
- **rush_att**: float — Intentos de acarreo.
- **snap_pct**: float — % de snaps.
- **team_pass_rate_neutral, team_pace**: float — Tendencias.
- **opp_vs_pass_epa, opp_vs_run_epa**: float — Fuerza rival.
- **vegas_total, vegas_spread**: float — Vegas.
- **weather_temp, weather_wind**: float — Clima.
- **outcome_yards**: float — Etiqueta.
