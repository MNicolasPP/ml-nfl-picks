from __future__ import annotations
import os, joblib, pandas as pd
from ..utils.io import load_table

def predict(input_path: str, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    df = load_table(input_path)
    ignore = {'game_id','player_id','player','team','opp','position','outcome_yards'}
    X = df.drop(columns=[c for c in ignore if c in df.columns])
    df['pred_yards'] = model.predict(X)
    return df
