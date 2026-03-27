import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd] + list(cwd.parents)
    for c in candidates:
        if (c / "materials" / "dataset_task3_exist2026").exists() and (
            c / "notebooks_shiyi"
        ).exists():
            return c / "notebooks_shiyi"
    if (cwd / "artifacts").exists() and (cwd / "entregables").exists():
        return cwd
    raise FileNotFoundError("Could not resolve notebooks_shiyi root.")


def export_json(sample_ids, preds, output_path: Path, test_case: str = "EXIST2026"):
    label_map = {0: "NO", 1: "YES"}
    rows = [
        {
            "test_case": test_case,
            "id": str(sid),
            "value": label_map[int(p)],
        }
        for sid, p in zip(sample_ids, preds)
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Export model predictions to JSON in predicciones/."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model .joblib paths.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to fusion_features.npz (optional).",
    )
    parser.add_argument(
        "--output-prefix",
        default="BeingChillingWeWillWin",
        help="Prefix used in output JSON filenames.",
    )
    parser.add_argument(
        "--test-case",
        default="EXIST2026",
        help="Value for test_case field in JSON rows.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()
    artifacts_dir = project_root / "artifacts"
    entregables_dir = project_root / "entregables"
    pred_dir = project_root / "predicciones"

    # Requirement: predicciones must be at the same hierarchy level as entregables.
    if entregables_dir.parent != pred_dir.parent:
        raise RuntimeError(
            "predicciones and entregables are not at the same hierarchy level."
        )

    pred_dir.mkdir(parents=True, exist_ok=True)

    features_path = (
        Path(args.features) if args.features else artifacts_dir / "fusion_features.npz"
    )
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    bundle = np.load(features_path, allow_pickle=True)
    X_all = bundle["X_fusion"].astype(np.float32)
    sample_ids = bundle["sample_ids"].astype(str)

    print("Using features:", features_path)
    print("Samples:", len(sample_ids), "| Feature dim:", X_all.shape[1])
    print("Output directory:", pred_dir)

    for model_path_str in args.models:
        model_path = Path(model_path_str)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        preds = model.predict(X_all)

        model_name = model_path.stem
        out_name = f"{args.output_prefix}_{model_name}.json"
        out_path = pred_dir / out_name
        export_json(sample_ids, preds, out_path, test_case=args.test_case)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
