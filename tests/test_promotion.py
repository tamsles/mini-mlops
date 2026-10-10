
import json
from types import SimpleNamespace

import pytest

from mini_mlops import promotion


def make_run(score: float):
    return SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(
            metrics={"macro_f1": score}
        ),
    )


def test_low_quality_candidate_cannot_replace_champion(
    tmp_path, monkeypatch
):
    # Existing production champion.
    path = tmp_path / "champion.json"

    old_champion = {
        "run_id": "old-run",
        "model_uri": "runs:/old-run/model",
    }

    path.write_text(
        json.dumps(old_champion),
        encoding="utf-8",
    )

    # New candidate is low quality.
    monkeypatch.setattr(
        promotion.mlflow,
        "get_run",
        lambda run_id: make_run(score=0.20),
    )

    # Promotion should reject BEFORE loading the model.
    def forbidden_load(uri):
        raise AssertionError(
            "Rejected candidate should never be loaded"
        )

    monkeypatch.setattr(
        promotion.mlflow.sklearn,
        "load_model",
        forbidden_load,
    )

    with pytest.raises(promotion.PromotionRejected):
        promotion.promote_candidate(
            run_id="bad-run",
            model_uri="runs:/bad-run/model",
            threshold=0.60,
            path=path,
        )

    # Champion must remain unchanged.
    actual = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert actual == old_champion


def test_good_candidate_can_be_promoted(
    tmp_path, monkeypatch
):
    path = tmp_path / "champion.json"

    monkeypatch.setattr(
        promotion.mlflow,
        "get_run",
        lambda run_id: make_run(score=0.85),
    )

    monkeypatch.setattr(
        promotion,
        "_source_run_id",
        lambda uri: "good-run",
    )

    class FakeModel:
        def predict(self, samples):
            return ["nlp", "cv"]

    monkeypatch.setattr(
        promotion.mlflow.sklearn,
        "load_model",
        lambda uri: FakeModel(),
    )

    monkeypatch.setattr(
        promotion.mlflow,
        "get_tracking_uri",
        lambda: "sqlite:///test.db",
    )

    result = promotion.promote_candidate(
        run_id="good-run",
        model_uri="runs:/good-run/model",
        threshold=0.60,
        path=path,
    )

    actual = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert actual["run_id"] == "good-run"
    assert actual["macro_f1"] == 0.85
    assert actual == result
