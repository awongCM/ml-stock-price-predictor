from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = "Close") -> None:
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        ...
