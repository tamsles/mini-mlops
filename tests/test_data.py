
import pandas as pd
import pytest

from mini_mlops.data import (
    load_data,
    validate_schema,
    split_data,
)


def test_load_data(sample_csv, sample_df):
    result = load_data(sample_csv)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        sample_df.reset_index(drop=True),
    )


def test_validate_schema_accepts_valid_data(sample_df):
    validate_schema(sample_df)


def test_validate_schema_rejects_missing_column(sample_df):
    invalid_df = sample_df.drop(columns=["label"])

    with pytest.raises(ValueError):
        validate_schema(invalid_df)


def test_split_preserves_all_rows(sample_df):
    train_df, test_df = split_data(
        sample_df,
        test_size=0.25,
        seed=42,
    )

    assert len(train_df) + len(test_df) == len(sample_df)


def test_split_has_no_overlap(sample_df):
    train_df, test_df = split_data(
        sample_df,
        test_size=0.25,
        seed=42,
    )

    train_texts = set(train_df["text"])
    test_texts = set(test_df["text"])

    assert train_texts.isdisjoint(test_texts)


def test_split_is_deterministic(sample_df):
    train_a, test_a = split_data(
        sample_df,
        test_size=0.25,
        seed=42,
    )

    train_b, test_b = split_data(
        sample_df,
        test_size=0.25,
        seed=42,
    )

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)



def test_load_data_calls_pandas(
    monkeypatch,
    sample_df,
    tmp_path,
):
    requested_paths = []

    def fake_read_csv(path, *args, **kwargs):
        requested_paths.append(path)
        return sample_df.copy()

    monkeypatch.setattr(
        "mini_mlops.data.pd.read_csv",
        fake_read_csv,
    )

    path = tmp_path / "fake.csv"

    result = load_data(path)

    assert requested_paths == [path]

    pd.testing.assert_frame_equal(
        result,
        sample_df,
    )
