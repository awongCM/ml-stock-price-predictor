import pandas as pd
import pytest

from app.models.lstm import LSTMModel, MIN_ROWS


def test_lstm_requires_minimum_rows():
    idx = pd.date_range("2024-01-01", periods=MIN_ROWS - 1, freq="D")
    df = pd.DataFrame({"Close": range(MIN_ROWS - 1)}, index=idx)
    model = LSTMModel(epochs=1, sequence_length=10)
    with pytest.raises(ValueError):
        model.fit(df)


@pytest.mark.slow
def test_lstm_predict_shape():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({"Close": (pd.Series(range(120)) + 100).astype(float)}, index=idx)
    model = LSTMModel(epochs=1, sequence_length=10)
    model.fit(df)
    preds = model.predict(df, horizon=3)
    assert len(preds) == 3
