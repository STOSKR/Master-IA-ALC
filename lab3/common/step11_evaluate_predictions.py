import argparse
import glob
import importlib
import os
from pathlib import Path

try:
    _pyevall_eval = importlib.import_module("pyevall.evaluation")
    _pyevall_metrics = importlib.import_module("pyevall.metrics.metricfactory")
    PyEvALLEvaluation = _pyevall_eval.PyEvALLEvaluation
    MetricFactory = _pyevall_metrics.MetricFactory
    HAS_PYEVALL = True
except ImportError:
    PyEvALLEvaluation = None
    MetricFactory = None
    HAS_PYEVALL = False

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTION_DIRS = [
    SCRIPT_DIR / "predicciones",
    SCRIPT_DIR / "predictions",
]


def resolve_predictions_dir(custom_dir: str | None = None) -> Path | None:
    if custom_dir:
        candidate = Path(custom_dir)
        if not candidate.is_absolute():
            candidate = (SCRIPT_DIR / candidate).resolve()
        return candidate

    for candidate in DEFAULT_PREDICTION_DIRS:
        if candidate.exists():
            return candidate

    return None


def _load_label_map(path: str | Path) -> dict[str, str]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(row["id"]): str(row["value"]).upper() for row in data}


def _local_binary_metrics(
    pred_path: str | Path, gold_path: str | Path
) -> tuple[float, float, int]:
    pred_map = _load_label_map(pred_path)
    gold_map = _load_label_map(gold_path)

    ids = [sid for sid in gold_map.keys() if sid in pred_map]
    if not ids:
        return 0.0, 0.0, 0

    y_true = [gold_map[sid] for sid in ids]
    y_pred = [pred_map[sid] for sid in ids]

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    f1_scores = []
    for label in ("NO", "YES"):
        tp = sum((t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum((t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum((t == label and p != label) for t, p in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1_scores.append(f1)

    f1_macro = sum(f1_scores) / len(f1_scores)
    return acc, f1_macro, len(ids)


def evaluate_all_predictions(predictions_dir: Path):
    gold_path = predictions_dir / "test_gold_pyevall.json"

    if not gold_path.exists():
        print(f"❌ Error: No se encontró el archivo gold en {gold_path}.")
        print(
            "Asegúrate de haber ejecutado antes el script step10_prepare_pyevall_test.py"
        )
        return

    # Buscar todos los JSON que empiecen por "Being" en la carpeta
    search_pattern = os.path.join(str(predictions_dir), "Being*.json")
    prediction_files = glob.glob(search_pattern)

    if not prediction_files:
        print(
            "⚠️ No se encontraron archivos de predicciones que empiecen por "
            f"'Being' en {predictions_dir}"
        )
        return

    print(f"📁 Carpeta de predicciones: {predictions_dir}")
    print(f"📁 Gold usado: {gold_path}")
    print(
        f"✅ Se encontraron {len(prediction_files)} archivos de predicciones para evaluar.\n"
    )

    if HAS_PYEVALL:
        test_eval = PyEvALLEvaluation()
        metrics = [MetricFactory.Accuracy.value, MetricFactory.FMeasure.value]
    else:
        print(
            "⚠️ pyevall no está instalado. Se usará evaluación local (Accuracy + F1-macro)."
        )
        print("   Para usar PyEvALL: pip install pyevall==0.1.78")
        print("")

    for preds_path in sorted(prediction_files):
        filename = os.path.basename(preds_path)
        print("=" * 60)
        print(f"📊 Evaluando: {filename}")
        print("=" * 60)

        try:
            if HAS_PYEVALL:
                # Evaluar usando pyevall
                report = test_eval.evaluate(preds_path, str(gold_path), metrics)
                report.print_report()
            else:
                acc, f1_macro, overlap = _local_binary_metrics(preds_path, gold_path)
                print(f"Registros evaluados (solape ID): {overlap}")
                if overlap == 0:
                    print("❌ No hay IDs en común entre predicción y gold.")
                else:
                    print(f"Accuracy: {acc:.6f}")
                    print(f"F1-macro: {f1_macro:.6f}")
            print("\n")
        except Exception as e:
            print(f"❌ Ocurrió un error al evaluar {filename}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalua JSON de predicciones con PyEvALL"
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help=(
            "Ruta de la carpeta con test_gold_pyevall.json y archivos Being*.json. "
            "Si no se indica, prueba primero ./predicciones y luego ./predictions "
            "relativas al script."
        ),
    )
    args = parser.parse_args()

    pred_dir = resolve_predictions_dir(args.predictions_dir)
    if pred_dir is None:
        print("❌ Error: No se encontró carpeta de predicciones.")
        print("Rutas probadas:")
        for candidate in DEFAULT_PREDICTION_DIRS:
            print(f" - {candidate}")
    else:
        evaluate_all_predictions(pred_dir)
