# Define and validate invariants


from dataclasses import dataclass

import pandas as pd


class DataContractError(ValueError):
    """Input dataset violates the training data contract."""


@dataclass(frozen=True)
class TrainingDataContract:
    allowed_labels: frozenset[str] = frozenset({
        "nlp",
        "cv",
        "mlops",
        "other",
    })

    min_rows: int = 16
    min_per_class: int = 2
    max_class_fraction: float = 0.70


DEFAULT_CONTRACT = TrainingDataContract()


def validate_training_data(
    df: pd.DataFrame,
    contract: TrainingDataContract = DEFAULT_CONTRACT,
) -> None:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    if (
        contract.min_rows < 1
        or contract.min_per_class < 1
        or not 0 < contract.max_class_fraction <= 1
    ):
        raise ValueError("Invalid data contract configuration")

    # 1. Required columns
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise DataContractError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 2. Missing values
    if df[["text", "label"]].isna().any().any():
        raise DataContractError(
            "Missing values in required columns"
        )

    # 3. Valid text
    valid_text = df["text"].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    )

    if not valid_text.all():
        raise DataContractError(
            "Invalid text: expected non-empty strings"
        )

    # 4. Valid label format
    valid_label_format = df["label"].map(
        lambda x: (
            isinstance(x, str)
            and bool(x)
            and x == x.strip()
        )
    )

    if not valid_label_format.all():
        raise DataContractError(
            "Invalid label format"
        )

    # 5. Allowed labels
    actual_labels = set(df["label"].unique())
    unknown_labels = actual_labels - contract.allowed_labels

    if unknown_labels:
        raise DataContractError(
            f"Unknown labels: {sorted(unknown_labels)}"
        )

    # 6. Duplicate text
    normalized_text = df["text"].str.strip().str.casefold()

    if normalized_text.duplicated().any():
        raise DataContractError(
            "Duplicate text detected"
        )

    # 7. Minimum dataset size
    if len(df) < contract.min_rows:
        raise DataContractError(
            f"Dataset too small: {len(df)} < {contract.min_rows}"
        )

    # 8. Class coverage
    counts = df["label"].value_counts().to_dict()

    insufficient = {
        label: counts.get(label, 0)
        for label in sorted(contract.allowed_labels)
        if counts.get(label, 0) < contract.min_per_class
    }

    if insufficient:
        raise DataContractError(
            f"Insufficient samples per class: {insufficient}"
        )

    # 9. Class distribution
    max_fraction = max(counts.values()) / len(df)

    if max_fraction > contract.max_class_fraction:
        raise DataContractError(
            f"Class imbalance too high: {max_fraction:.2%}"
        )
