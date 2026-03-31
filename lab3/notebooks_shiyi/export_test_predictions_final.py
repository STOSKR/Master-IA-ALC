import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta predicciones finales sobre el conjunto test oficial a partir "
            "de JSON Being*.json generados por notebooks."
        )
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=["entregables", "prediction", "predicciones"],
        help="Directorios de entrada donde buscar archivos Being*.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="prediccion_final",
        help="Directorio de salida para JSON filtrados al set test.",
    )
    parser.add_argument(
        "--training-json",
        default="../materials/dataset_task3_exist2026/EXIST2026_training.json",
        help="Ruta a EXIST2026_training.json.",
    )
    parser.add_argument(
        "--test-json",
        default="../materials/dataset_task3_exist2026/test.json",
        help="Ruta a test.json.",
    )
    parser.add_argument(
        "--test-case",
        default="EXIST2026",
        help="Valor del campo test_case en los JSON de salida.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Si se activa, solo exporta archivos con cobertura completa del test.",
    )
    return parser.parse_args()


def load_target_test_ids(training_json: Path, test_json: Path) -> list[str]:
    with training_json.open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    with test_json.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    target_ids: list[str] = []
    seen: set[str] = set()

    # Conserva el orden del test.json y mapea key -> id_EXIST.
    for key in test_data.keys():
        item = train_data.get(key)
        if item is None:
            continue
        sid = str(item.get("id_EXIST", key))
        if sid not in seen:
            seen.add(sid)
            target_ids.append(sid)

    return target_ids


def read_prediction_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Formato invalido (no lista) en: {path}")
    return data


def normalize_label(value: str) -> str:
    v = str(value).strip().upper()
    if v not in {"YES", "NO"}:
        raise ValueError(f"Label invalida: {value}")
    return v


def filter_to_test(
    rows: list[dict], target_ids: list[str], test_case: str
) -> tuple[list[dict], int]:
    # Si hay ids duplicados, prevalece el ultimo valor encontrado.
    pred_map: dict[str, str] = {}
    for row in rows:
        sid = str(row.get("id", "")).strip()
        if not sid:
            continue
        pred_map[sid] = normalize_label(row.get("value", ""))

    out_rows: list[dict] = []
    missing = 0
    for sid in target_ids:
        value = pred_map.get(sid)
        if value is None:
            missing += 1
            continue
        out_rows.append({"test_case": test_case, "id": sid, "value": value})

    return out_rows, missing


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    source_dirs = [
        (Path(d) if Path(d).is_absolute() else (script_dir / d).resolve())
        for d in args.source_dirs
    ]
    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else (script_dir / args.output_dir).resolve()
    )
    training_json = (
        Path(args.training_json)
        if Path(args.training_json).is_absolute()
        else (script_dir / args.training_json).resolve()
    )
    test_json = (
        Path(args.test_json)
        if Path(args.test_json).is_absolute()
        else (script_dir / args.test_json).resolve()
    )

    if not training_json.exists():
        raise FileNotFoundError(f"No existe training_json: {training_json}")
    if not test_json.exists():
        raise FileNotFoundError(f"No existe test_json: {test_json}")

    target_ids = load_target_test_ids(training_json, test_json)
    if not target_ids:
        raise RuntimeError("No se pudieron resolver IDs objetivo del test.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Recolecta archivos Being*.json por orden de source_dirs.
    # Si el mismo nombre aparece en varios directorios, prevalece el ultimo.
    selected_files: dict[str, Path] = {}
    for sdir in source_dirs:
        if not sdir.exists():
            print(f"WARNING: directorio no existe, se omite: {sdir}")
            continue
        for path in sorted(sdir.glob("Being*.json")):
            selected_files[path.name] = path

    if not selected_files:
        print("ERROR: no se encontraron archivos Being*.json en source_dirs")
        return 1

    summary_rows: list[dict[str, str | int]] = []
    exported = 0

    for name in sorted(selected_files.keys()):
        src_path = selected_files[name]
        try:
            rows = read_prediction_rows(src_path)
            test_rows, missing = filter_to_test(rows, target_ids, args.test_case)
            overlap = len(test_rows)
            total_target = len(target_ids)
            complete = missing == 0

            status = "exported"
            if args.require_complete and not complete:
                status = "skipped_incomplete"
            elif overlap == 0:
                status = "skipped_no_overlap"

            if status == "exported":
                out_path = output_dir / name
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(test_rows, f, ensure_ascii=False)
                exported += 1

            summary_rows.append(
                {
                    "file": name,
                    "source": str(src_path),
                    "target_test_ids": total_target,
                    "overlap": overlap,
                    "missing": missing,
                    "source_rows": len(rows),
                    "exported_rows": overlap if status == "exported" else 0,
                    "status": status,
                }
            )
            print(
                f"{name}: overlap={overlap}/{total_target} missing={missing} "
                f"status={status}"
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "file": name,
                    "source": str(src_path),
                    "target_test_ids": len(target_ids),
                    "overlap": 0,
                    "missing": len(target_ids),
                    "source_rows": 0,
                    "exported_rows": 0,
                    "status": f"error: {exc}",
                }
            )
            print(f"ERROR procesando {name}: {exc}")

    summary_path = output_dir / "summary_export_test_predictions.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "source",
                "target_test_ids",
                "overlap",
                "missing",
                "source_rows",
                "exported_rows",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("------------------------------------------")
    print(f"Target test ids: {len(target_ids)}")
    print(f"Files processed: {len(summary_rows)}")
    print(f"Files exported : {exported}")
    print(f"Output dir     : {output_dir}")
    print(f"Summary CSV    : {summary_path}")
    print("------------------------------------------")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
