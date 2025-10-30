# Dashboard — Requerimientos mínimos

## Vistas
1. **Picks de la semana**: jugador, equipo, rival, línea, predicción, Prob(Over), edge, estatus.
2. **Por juego**: filtro por game_id/semana.
3. **Explicabilidad**: SHAP (top features) por jugador.

## Filtros
- Equipo, posición, rango de línea, prob mínima, edge mínimo.

## Reglas de publicación
- Prob(Over) ≥ 0.58 y edge ≥ 0.08; excluir por lesión/clima extremo.
