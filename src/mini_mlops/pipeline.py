
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mini_mlops.contracts import validate_training_data
from mini_mlops.lineage import sha256_file
from mini_mlops.quality import ModelQualityResult, require_quality
from mini_mlops.training import train_workload


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mini-MLOps local pipeline"
    )
    parser.add_argument(
        "--data", default="data/raw/projects.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--min-macro-f1", type=float, default=0.60
    )

    args = parser.parse_args()

    if not 0 <= args.min_macro_f1 <= 1:
        parser.error("--min-macro-f1 must be in [0, 1]")

    # Make relative paths and MLflow state consistent.
    os.chdir(ROOT)

    data_path = Path(args.data).resolve()

    try:
        # Step 1: Validate the actual training dataset.
        print("\n[1/4] VALIDATE", flush=True)

        df = pd.read_csv(data_path)
        validate_training_data(df)

        expected_sha = sha256_file(data_path)
        print(f"VALIDATION PASS: {len(df)} rows", flush=True)

        # Step 2: Run tests; stop on non-zero exit.
        print("\n[2/4] TEST", flush=True)

        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=ROOT,
            check=True,
        )
        print("TEST PASS", flush=True)

        # Step 3: Train one new candidate.
        print("\n[3/4] TRAIN", flush=True)

        result = train_workload(
            data_path=data_path,
            seed=args.seed,
            test_size=args.test_size,
        )

        if not result.run_id or not result.model_uri:
            raise RuntimeError(
                "Training returned missing model identity"
            )

        # Check that training used the validated data.
        if sha256_file(data_path) != expected_sha:
            raise RuntimeError("Dataset changed during pipeline")

        if result.data_sha256 != expected_sha:
            raise RuntimeError("Training data SHA256 mismatch")

        print(f"TRAIN PASS: run_id={result.run_id}", flush=True)
        print(f"model_uri={result.model_uri}", flush=True)

        # Step 4: Enforce quality on THIS training result.
        print("\n[4/4] QUALITY GATE", flush=True)

        score = float(result.macro_f1)
        passed = (
            math.isfinite(score)
            and 0 <= score <= 1
            and score >= args.min_macro_f1
        )

        quality = ModelQualityResult(
            macro_f1=score,
            threshold=args.min_macro_f1,
            passed=passed,
        )

        require_quality(quality)

        print(f"GATE PASS: macro_f1={score:.4f}", flush=True)
        print("\nPIPELINE SUCCESS", flush=True)
        return 0

    except Exception as exc:
        print(
            f"\nPIPELINE FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
