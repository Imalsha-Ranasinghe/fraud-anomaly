from src.data import load_transactions
from src.train import split_by_time, train


def test_split_has_no_time_overlap():
    train_df, test_df = split_by_time(load_transactions())
    assert train_df["Time"].max() <= test_df["Time"].min()

def test_split_keeps_all_rows():
    df = load_transactions()
    train_df, test_df = split_by_time(df, test_frac=0.2)
    assert len(train_df) + len(test_df) == len(df)

def test_train_beats_random_baseline():
    _, metrics = train()
    assert metrics["pr_auc"] > 0.1