#!/bin/bash
#SBATCH -p long
#SBATCH --cpus-per-task=8
#SBATCH --job-name=run
#SBATCH --mem=32G
#SBATCH --gres=shard:4
#SBATCH -o logs/%j.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Perfil de ejecucion:
# - proposals (por defecto): ejecuta solo notebooks nuevos de propuestas.
# - discovery: escanea carpetas (comportamiento legacy).
# Para generar predicciones por idioma (ES/EN/ES_EN) se recomienda proposals.
RUN_PROFILE=${RUN_PROFILE:-"proposals"}

# Lista explicita para perfil proposals.
# Puedes sobreescribir al lanzar:
#   PROPOSAL_NOTEBOOKS="EN/03_train_classifier_en.ipynb ES/02_train_classifier_es.ipynb ES_EN/04_train_classifier_es_en.ipynb" sbatch run.sh
PROPOSAL_NOTEBOOKS=${PROPOSAL_NOTEBOOKS:-"EN/03_train_classifier_en.ipynb ES/02_train_classifier_es.ipynb ES_EN/04_train_classifier_es_en.ipynb"}

# Opciones usadas solo en perfil discovery.
# Puedes sobreescribir al lanzar:
#   RUN_PROFILE=discovery LANG_FOLDERS="ES EN ES_EN" sbatch run.sh
LANG_FOLDERS=${LANG_FOLDERS:-"ES EN ES_EN"}

# Si quieres incluir tambien notebooks de raiz en discovery, activa esto en 1.
# Ejemplo: RUN_PROFILE=discovery INCLUDE_ROOT_NOTEBOOKS=1 sbatch run.sh
INCLUDE_ROOT_NOTEBOOKS=${INCLUDE_ROOT_NOTEBOOKS:-0}

ENTREGABLES_DIR="entregables"
PREDICTION_DIR="prediction"
FINAL_PREDICTION_DIR=${FINAL_PREDICTION_DIR:-"prediccion_final"}
EXPORT_TEST_FINAL=${EXPORT_TEST_FINAL:-1}
FINAL_REQUIRE_COMPLETE=${FINAL_REQUIRE_COMPLETE:-0}
SOURCE_JSON_DIRS=("$ENTREGABLES_DIR" "predicciones" "$FINAL_PREDICTION_DIR")

mkdir -p "$ENTREGABLES_DIR" "$PREDICTION_DIR" "$FINAL_PREDICTION_DIR"

declare -a NOTEBOOK_QUEUE=()

if [ "$RUN_PROFILE" = "proposals" ]; then
    for nb_file in $PROPOSAL_NOTEBOOKS; do
        NOTEBOOK_QUEUE+=("$nb_file")
    done
else
    if [ "$INCLUDE_ROOT_NOTEBOOKS" = "1" ]; then
        while IFS= read -r nb_file; do
            NOTEBOOK_QUEUE+=("$nb_file")
        done < <(find . -maxdepth 1 -type f -name "*.ipynb" -printf "%f\n" | sort)
    fi

    for LANG_DIR in $LANG_FOLDERS; do
        if [ -d "$LANG_DIR" ]; then
            while IFS= read -r nb_file; do
                NOTEBOOK_QUEUE+=("$nb_file")
            done < <(find "$LANG_DIR" -maxdepth 1 -type f -name "*.ipynb" | sort)
        else
            echo "WARNING: carpeta de idioma no encontrada, se omite: $LANG_DIR"
        fi
    done
fi

if [ ${#NOTEBOOK_QUEUE[@]} -eq 0 ]; then
    echo "ERROR: no hay notebooks para ejecutar (cola vacia)."
    exit 1
fi

echo "Total notebooks en cola: ${#NOTEBOOK_QUEUE[@]}"
printf ' - %s\n' "${NOTEBOOK_QUEUE[@]}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate RFA2526pt

# Configure CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure we use only the GPU assigned by SLURM
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi

# Print GPU memory info
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# Ejecutar cada notebook secuencialmente
for NOTEBOOK in "${NOTEBOOK_QUEUE[@]}"; do
    if [ ! -f "$NOTEBOOK" ]; then
        echo "WARNING: notebook no encontrado, se omite: $NOTEBOOK"
        continue
    fi

    echo "=========================================="
    echo "Starting execution of: $NOTEBOOK"
    echo "=========================================="
    
    export NB_IN="$NOTEBOOK"
    export ENTREGABLES_DIR
    
    python << 'EOF'
import os, nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import torch
import gc

input_file = os.environ['NB_IN']
out_dir = os.environ.get('ENTREGABLES_DIR', 'entregables')

os.makedirs(out_dir, exist_ok=True)

# Use a safe filename so paths like ES/xx.ipynb do not create nested folders in entregables.
safe_input_name = input_file.replace(os.sep, '__').replace('/', '__')

# Prepare output filename
base_name = safe_input_name.replace('.ipynb', '_out.ipynb')
output_file = os.path.join(out_dir, base_name)

counter = 1
root, ext = os.path.splitext(base_name)
while os.path.exists(output_file):
    output_file = os.path.join(out_dir, f"{root}_{counter}{ext}")
    counter += 1

# Prepare error filename
error_base_name = safe_input_name.replace('.ipynb', '_error.ipynb')
error_output_file = os.path.join(out_dir, error_base_name)

error_counter = 1
error_root, error_ext = os.path.splitext(error_base_name)
while os.path.exists(error_output_file):
    error_output_file = os.path.join(out_dir, f"{error_root}_{error_counter}{error_ext}")
    error_counter += 1

with open(input_file) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
exec_dir = os.path.dirname(input_file) or './'

try:
    ep.preprocess(nb, {'metadata': {'path': exec_dir}})
    # Save successful execution
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully saved to: {output_file}")
    # Clean up memory after successful execution
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("Memory cleaned after successful execution")
except Exception as e:
    # Save notebook with error state
    with open(error_output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Error occurred. Notebook saved to: {error_output_file}")
    print(f"Error: {e}")
    # Continue with next notebook even if this one fails
    exit(1)
finally:
    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("Memory cleaned")
EOF
    
    if [ $? -ne 0 ]; then
        echo "WARNING: $NOTEBOOK execution failed, continuing with next notebook..."
    else
        echo "Notebook completado: $NOTEBOOK"
    fi

    # Copiar predicciones JSON acumuladas a prediction tras cada notebook.
    copied_any=0
    for src_dir in "${SOURCE_JSON_DIRS[@]}"; do
        if [ ! -d "$src_dir" ]; then
            continue
        fi
        for json_file in "$src_dir"/*.json; do
            if [ -e "$json_file" ]; then
                cp -f "$json_file" "$PREDICTION_DIR"/
                copied_any=1
            fi
        done
    done

    if [ "$copied_any" -eq 1 ]; then
        total_json=$(ls -1 "$PREDICTION_DIR"/*.json 2>/dev/null | wc -l)
        echo "JSON sincronizados en $PREDICTION_DIR (total: $total_json)"
    else
        echo "No hay JSON nuevos para sincronizar a $PREDICTION_DIR"
    fi
    
    echo ""
done

echo "=========================================="
echo "All notebooks processed!"
echo "Prediction dir: $PREDICTION_DIR"

if [ "$EXPORT_TEST_FINAL" = "1" ]; then
    echo "=========================================="
    echo "Exporting test-set final predictions"
    echo "Output dir: $FINAL_PREDICTION_DIR"
    echo "=========================================="

    REQUIRE_FLAG=""
    if [ "$FINAL_REQUIRE_COMPLETE" = "1" ]; then
        REQUIRE_FLAG="--require-complete"
    fi

    # Construye JSON finales sobre IDs del test oficial (mismo universo para comparar modelos).
    python export_test_predictions_final.py \
        --source-dirs "$ENTREGABLES_DIR" "$PREDICTION_DIR" predicciones "$FINAL_PREDICTION_DIR" \
        --output-dir "$FINAL_PREDICTION_DIR" \
        --training-json "../materials/dataset_task3_exist2026/EXIST2026_training.json" \
        --test-json "../materials/dataset_task3_exist2026/test.json" \
        $REQUIRE_FLAG

    if [ $? -ne 0 ]; then
        echo "WARNING: final test prediction export failed"
    else
        echo "Final test predictions exported to: $FINAL_PREDICTION_DIR"
    fi
fi

echo "=========================================="