import pandas as pd

from app.models.sma import SMAModel


def test_sma_predict_constant_series():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Close": [100.0] * 30}, index=idx)
    model = SMAModel(window=5)
    model.fit(df)
    preds = model.predict(df, horizon=3)
    assert len(preds) == 3
    assert preds.iloc[0] == 100.0
