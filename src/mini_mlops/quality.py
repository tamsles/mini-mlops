
from dataclasses import dataclass
from typing import Sequence

from sklearn.metrics import f1_score


LABELS = ("cv", "mlops", "nlp", "other")


class ModelQualityError(ValueError):
    """Candidate model fails the quality requirement."""


@dataclass(frozen=True)
class ModelQualityResult:
    macro_f1: float
    threshold: float
    passed: bool


def evaluate_quality(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    min_macro_f1: float = 0.60,
) -> ModelQualityResult:

    y_true = list(y_true)
    y_pred = list(y_pred)

    if not 0.0 <= min_macro_f1 <= 1.0:
        raise ValueError("Invalid macro-F1 threshold")

    if not y_true or len(y_true) != len(y_pred):
        raise ValueError(
            "Evaluation arrays must be non-empty and equal in length"
        )

    allowed = set(LABELS)

    if set(y_true) != allowed:
        raise ValueError(
            "Evaluation set must contain all four classes"
        )

    if not set(y_pred).issubset(allowed):
        raise ValueError("Predictions contain unknown labels")

    score = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
    )

    return ModelQualityResult(
        macro_f1=score,
        threshold=min_macro_f1,
        passed=score >= min_macro_f1,
    )


def require_quality(result: ModelQualityResult) -> None:
    if not result.passed:
        raise ModelQualityError(
            f"Model rejected: macro-F1={result.macro_f1:.3f}, "
            f"required={result.threshold:.3f}"
        )
