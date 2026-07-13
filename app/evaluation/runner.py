from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.backtest import run_backtest
from app.evaluation.metrics import mape, rmse
from app.evaluation.split import train_holdout_split
from app.models.base import BaseModel


def evaluate_model(model: BaseModel, df: pd.DataFrame, horizon: int = 1) -> dict:
    train, holdout = train_holdout_split(df)
    model.fit(train)

    preds = []
    actuals = []
    rolling = train.copy()
    for i in range(len(holdout)):
        prediction = model.predict(rolling, horizon=1)
        preds.append(float(prediction.iloc[0]))
        actuals.append(float(holdout["Close"].iloc[i]))
        rolling = pd.concat([rolling, holdout.iloc[[i]]])

    pred_arr = np.array(preds)
    act_arr = np.array(actuals)
    bt = run_backtest(holdout["Close"], pd.Series(pred_arr, index=holdout.index))
    return {
        "rmse": rmse(act_arr, pred_arr),
        "mape": mape(act_arr, pred_arr),
        **bt,
    }
