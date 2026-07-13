import pandas as pd

from app.evaluation.split import train_holdout_split


def test_train_holdout_split():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": range(10)}, index=idx)
    train, holdout = train_holdout_split(df, holdout_ratio=0.2)
    assert len(train) == 8
    assert len(holdout) == 2
