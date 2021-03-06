import sys,os
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fire import Fire
from pathlib import Path

from src.utils import load_splits

def stratified_k_fold(
    description,
    n_folds,
    stratified_by,
    random_state=42
):
    folds = []

    X = description
    y = description[stratified_by]

    stratifier = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for _, test_indexes in stratifier.split(X, y):
        folds.append(X.iloc[test_indexes])

    folds = [pd.DataFrame(fold, columns=description.columns) for fold in folds]
    return folds


def main(
    folds_folder,
    mode,
    n_folds=5,
):
    assert mode in ["create", "test"], f"No such mode as: {mode}"

    if mode == "create":

        n_folds = 5
        df_fn = Path("data/train.csv")
        df = pd.read_csv(df_fn)

        df = df.sample(frac=1).reset_index(drop=True)
        folds = stratified_k_fold(df, n_folds, "label", random_state=42)

        folds_folder = Path(folds_folder)
        for idx, fold in enumerate(folds):
            fold_filename =  folds_folder / f"fold_{idx}.csv"
            fold.to_csv(fold_filename, index=False)

    elif mode == "test":
        folds_folder = Path(folds_folder)
        for i in range(n_folds):
            tr_df, val_df = load_splits(folds_folder, val_folds=i)
            print(f"fold: {i}")
            print(f"train split: {tr_df.shape}, target: {tr_df['label'].sum() / len(tr_df)}")
            print(f"val split: {val_df.shape}, target: {val_df['label'].sum() / len(val_df)}")
            print()


if __name__ == "__main__":
    Fire(main)

