from __future__ import annotations
import os, pandas as pd, pathlib

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def load_table(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def save_table(df, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
