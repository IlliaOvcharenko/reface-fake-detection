import numpy as np
import pandas as pd

from pathlib import Path
from fire import Fire
from math import ceil


def merge_dataframes(dfs):
    df = dfs[0]
    for idx in range(1, len(dfs)):
        df = pd.merge(
            df,
            dfs[idx],
            on="filename",
            suffixes=(None, f"_fold_{idx}")
        )
    df = df.rename(columns={"label": "label_fold_0"})
    return df


def main(
        preds_folder,
        save_fn,
        merge_strategy="vote",
):
    assert merge_strategy in ["vote", ], \
           f"There is no such merge strateguy available: {merge_strategy}"

    preds_folder = Path(preds_folder)
    preds_filenames = list(sorted(preds_folder.glob("*.csv")))
    preds_dfs = [pd.read_csv(fn) for fn in preds_filenames]

    df = merge_dataframes(preds_dfs)

    label_columns = [c for c in df.columns if "label" in c]
    if merge_strategy == "vote":
        result = (df[label_columns].mean(axis=1) >= 0.5).astype(int)

        # checks
        for c in label_columns:
            print(f"pred: {c}, zero rate: {df[c].sum() / len(df)}")
        print(f"result zero rate: {result.sum() / len(df)}")

        df = df.drop(columns=label_columns)
        df["label"] = result

    df.to_csv(save_fn, index=False)


if __name__ == "__main__":
    Fire(main)

