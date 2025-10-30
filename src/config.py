from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    data: Dict[str, Any]
    def get(self, path: str, default=None):
        cur = self.data
        for key in path.split('.'):
            if key in cur:
                cur = cur[key]
            else:
                return default
        return cur

def load_config(path: str = "configs/default.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Config(d)
