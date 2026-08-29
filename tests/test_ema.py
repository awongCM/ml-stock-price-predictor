import pandas as pd

from app.models.ema import EMAModel


def test_ema_predict_shape():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Close": [float(i) for i in range(30)]}, index=idx)
    model = EMAModel(span=5)
    model.fit(df)
    preds = model.predict(df, horizon=3)
    assert len(preds) == 3
