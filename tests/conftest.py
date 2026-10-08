
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    templates = {
        "nlp": "transformer language model for text classification",
        "cv": "image recognition using convolutional networks",
        "mlops": "model deployment tracking and monitoring pipeline",
        "other": "database indexing and operating system performance",
    }

    rows = []

    for label, text in templates.items():
        for i in range(6):
            rows.append({
                "text": f"{text} project number {i}",
                "label": label,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv(sample_df, tmp_path):
    path = tmp_path / "projects.csv"

    sample_df.to_csv(path, index=False)

    return path
