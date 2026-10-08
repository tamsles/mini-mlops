from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mini_mlops.contracts import validate_training_data

REQUIRED_COLUMNS = {"text", "label"}

def load_data(path : Path) -> pd.DataFrame:
    return pd.read_csv(path)
    """Load dataset from a CSV file."""


def validate_schema(df: pd.DataFrame) -> None:
    """Perform basic schema validation."""
    validate_training_data(df)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["text"].isna().any():
        raise ValueError("Column 'text' contains missing values.")

    if df["label"].isna().any():
        raise ValueError("Column 'label' contains missing values.")


def split_data(
    df: pd.DataFrame,
    test_size : float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    return train_df, test_df