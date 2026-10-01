"""Download the credit card fraud dataset from OpenML and save it as data/creditcard.csv.

The dataset is the well-known "Credit Card Fraud Detection" data released by the
Machine Learning Group of ULB (Universite Libre de Bruxelles). It contains 284,807
European card transactions from September 2013, of which 492 are fraud.

Run from the project root:
    python src/download_data.py
"""

from pathlib import Path

from sklearn.datasets import fetch_openml

OPENML_DATA_ID = 1597
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def main():
    if DATA_PATH.exists():
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading dataset from OpenML (about 150 MB, this can take a few minutes)...")
    dataset = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    df = dataset.frame
    df["Class"] = df["Class"].astype(int)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows and {df.shape[1]} columns to {DATA_PATH}")


if __name__ == "__main__":
    main()
