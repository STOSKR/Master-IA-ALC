import os
import glob
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory

PREDICTIONS_DIR = "./predictions/"
GOLD_PATH = os.path.join(PREDICTIONS_DIR, "test_gold_pyevall.json")

def evaluate_all_predictions():
    if not os.path.exists(GOLD_PATH):
        print(f"❌ Error: No se encontró el archivo gold en {GOLD_PATH}.")
        print("Asegúrate de haber ejecutado antes el script step10_prepare_pyevall_test.py")
        return

    # Buscar todos los JSON que empiecen por "Being" en la carpeta
    search_pattern = os.path.join(PREDICTIONS_DIR, "Being*.json")
    prediction_files = glob.glob(search_pattern)

    if not prediction_files:
        print(f"⚠️ No se encontraron archivos de predicciones que empiecen por 'Being' en {PREDICTIONS_DIR}")
        return

    print(f"✅ Se encontraron {len(prediction_files)} archivos de predicciones para evaluar.\n")

    test_eval = PyEvALLEvaluation()
    metrics = [MetricFactory.Accuracy.value, MetricFactory.FMeasure.value]

    for preds_path in sorted(prediction_files):
        filename = os.path.basename(preds_path)
        print("="*60)
        print(f"📊 Evaluando: {filename}")
        print("="*60)
        
        try:
            # Evaluar usando pyevall
            report = test_eval.evaluate(preds_path, GOLD_PATH, metrics)
            report.print_report()
            print("\n")
        except Exception as e:
            print(f"❌ Ocurrió un error al evaluar {filename}: {e}\n")

if __name__ == '__main__':
    evaluate_all_predictions()
