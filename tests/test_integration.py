
import pytest

from mini_mlops.data import (
    load_data,
    validate_schema,
    split_data,
)

from mini_mlops.model import build_model


@pytest.mark.integration
def test_core_pipeline_end_to_end(sample_csv):
    # Step 1: Load
    df = load_data(sample_csv)

    # Step 2: Validate
    validate_schema(df)

    # Step 3: Split
    train_df, test_df = split_data(
        df,
        test_size=0.25,
        seed=42,
    )

    # Step 4: Build and fit
    model = build_model()

    model.fit(
        train_df["text"],
        train_df["label"],
    )

    # Step 5: Predict on held-out samples
    predictions = model.predict(test_df["text"])

    # Step 6: Verify interface contracts
    assert len(predictions) == len(test_df)

    assert set(predictions).issubset(
        set(train_df["label"])
    )

    assert len(test_df) > 0
