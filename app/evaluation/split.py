import pandas as pd


def train_holdout_split(
    df: pd.DataFrame,
    holdout_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1")
    split_idx = int(len(df) * (1 - holdout_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
