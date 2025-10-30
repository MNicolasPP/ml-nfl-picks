from __future__ import annotations
import pandas as pd
from ..config import load_config
from ..utils.io import load_table, ensure_dir

def load_sources():
    cfg = load_config()
    paths = cfg.get('data_sources')
    tables = {k: load_table(v) for k, v in paths.items()}
    return tables
