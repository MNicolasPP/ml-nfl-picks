from __future__ import annotations
import os, joblib, json
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from ..config import load_config
from ..utils.io import load_table

def train_model(input_path: str, out_dir: str):
    cfg = load_config()
    os.makedirs(out_dir, exist_ok=True)
    df = load_table(input_path).dropna(subset=['outcome_yards'])

    target = 'outcome_yards'
    ignore = {'game_id','player_id','player','team','opp','position'}
    X = df.drop(columns=[target] + [c for c in ignore if c in df.columns])
    y = df[target].values

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    preferred = cfg.get('project.preferred_model','lightgbm')
    if preferred == 'xgboost':
        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=cfg.get('project.random_seed',42))
    else:
        model = LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, random_state=cfg.get('project.random_seed',42))

    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y[train_idx])
        p = model.predict(X.iloc[val_idx])
        maes.append(mean_absolute_error(y[val_idx], p))

    holdout_mae = float(np.mean(maes))

    model.fit(X, y)
    joblib.dump(model, os.path.join(out_dir, 'model.joblib'))

    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'mae_cv': holdout_mae}, f, indent=2)

    return holdout_mae
