from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"

def load_transactions() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = fetch_openml(data_id=1597, as_frame=True).frame
    df["Class"] = df["Class"].astype(int)
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df