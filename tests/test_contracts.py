
import pandas as pd
import pytest
from dataclasses import replace

from mini_mlops.contracts import (
    DataContractError,
    TrainingDataContract,
    validate_training_data,
)


def test_valid_dataset(sample_df):
    validate_training_data(sample_df)


def test_validation_does_not_modify_data(sample_df):
    original = sample_df.copy(deep=True)

    validate_training_data(sample_df)

    pd.testing.assert_frame_equal(sample_df, original)


@pytest.mark.parametrize("missing_column", ["text", "label"])
def test_missing_required_column(sample_df, missing_column):
    invalid = sample_df.drop(columns=[missing_column])

    with pytest.raises(
        DataContractError,
        match="Missing required columns",
    ):
        validate_training_data(invalid)

@pytest.mark.parametrize("bad_text", ["", "   ", 123])
def test_invalid_text(sample_df, bad_text):
    invalid = sample_df.copy()

    invalid["text"] = invalid["text"].astype(object)
    invalid.loc[0, "text"] = bad_text

    with pytest.raises(
        DataContractError,
        match="Invalid text",
    ):
        validate_training_data(invalid)


def test_duplicate_text(sample_df):
    invalid = sample_df.copy()

    invalid.loc[1, "text"] = invalid.loc[0, "text"]

    with pytest.raises(
        DataContractError,
        match="Duplicate text",
    ):
        validate_training_data(invalid)


def test_duplicate_after_normalization(sample_df):
    invalid = sample_df.copy()

    invalid.loc[1, "text"] = (
        "  " + invalid.loc[0, "text"].upper() + "  "
    )

    with pytest.raises(
        DataContractError,
        match="Duplicate text",
    ):
        validate_training_data(invalid)


def test_conflicting_labels_for_same_text(sample_df):
    invalid = sample_df.copy()

    # Row 0 is nlp, row 6 is cv in the fixture.
    invalid.loc[6, "text"] = invalid.loc[0, "text"]

    with pytest.raises(
        DataContractError,
        match="Duplicate text",
    ):
        validate_training_data(invalid)



def test_dataset_too_small(sample_df):
    invalid = sample_df.iloc[:10].copy()

    with pytest.raises(
        DataContractError,
        match="Dataset too small",
    ):
        validate_training_data(invalid)


def test_missing_class(sample_df):
    invalid = sample_df[
        sample_df["label"] != "nlp"
    ].copy()

    with pytest.raises(
        DataContractError,
        match="Insufficient samples per class",
    ):
        validate_training_data(invalid)


def test_rare_class(sample_df):
    nlp_indices = sample_df[
        sample_df["label"] == "nlp"
    ].index[:5]

    invalid = sample_df.drop(index=nlp_indices)

    with pytest.raises(
        DataContractError,
        match="Insufficient samples per class",
    ):
        validate_training_data(invalid)


@pytest.fixture
def imbalanced_df(sample_df):
    # Keep two samples per original class.
    base = sample_df.groupby(
        "label", sort=False
    ).head(2)

    # Add twenty distinct NLP samples.
    extra = pd.DataFrame([
        {
            "text": f"additional nlp document {i}",
            "label": "nlp",
        }
        for i in range(20)
    ])

    return pd.concat(
        [base, extra],
        ignore_index=True,
    )


def test_imbalanced_dataset(imbalanced_df):
    with pytest.raises(
        DataContractError,
        match="Class imbalance too high",
    ):
        validate_training_data(imbalanced_df)


def test_configurable_contract(imbalanced_df):
    strict = TrainingDataContract()

    relaxed = replace(
        strict,
        max_class_fraction=0.90,
    )

    validate_training_data(
        imbalanced_df,
        contract=relaxed,
    )
