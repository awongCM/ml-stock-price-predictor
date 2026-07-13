"""Forecast models."""

from app.models.ema import EMAModel
from app.models.lstm import LSTMModel
from app.models.sma import SMAModel

MODELS = {
    "sma": SMAModel,
    "ema": EMAModel,
    "lstm": LSTMModel,
}


def get_model(name: str):
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model {name}. Choose from {list(MODELS)}")
    return MODELS[key]()
