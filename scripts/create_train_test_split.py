import sys,os
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.model_selection import train_test_split

from fire import Fire
from pathlib import Path


def main():
    df_fn = Path("data/train.csv")
    df = pd.read_csv(df_fn)
    train_df, val_df = train_test_split(df,
                                        test_size=0.2,
                                        shuffle=True,
                                        random_state=42,
                                        stratify=df["label"])
    # print(train_df.shape, val_df.shape)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    save_folder = Path("data/splits")
    save_folder.mkdir()

    train_df.to_csv(save_folder / "train.csv", index=False)
    val_df.to_csv(save_folder / "val.csv", index=False)

if __name__ == "__main__":
    Fire(main)

