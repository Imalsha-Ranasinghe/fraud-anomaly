from pathlib import Path

import pandas as pd

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def load_transactions() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = pd.read_csv(DATA_URL)
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df