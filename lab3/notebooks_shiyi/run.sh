#!/bin/bash
#SBATCH -p long
#SBATCH --cpus-per-task=8
#SBATCH --job-name=run
#SBATCH --mem=32G
#SBATCH --gres=shard:6
#SBATCH -o logs/%j.log

# Modo recomendado: ejecutar todos los notebooks de carpetas de idioma.
# Puedes sobreescribir al lanzar:
#   LANG_FOLDERS="ES EN ES_EN" sbatch run.sh
LANG_FOLDERS=${LANG_FOLDERS:-"ES EN ES_EN"}

# Si quieres incluir tambien notebooks de raiz, activa esto en 1.
# Ejemplo: INCLUDE_ROOT_NOTEBOOKS=1 sbatch run.sh
INCLUDE_ROOT_NOTEBOOKS=${INCLUDE_ROOT_NOTEBOOKS:-0}

# Lista de notebooks de raiz (solo se usa si INCLUDE_ROOT_NOTEBOOKS=1).
ROOT_NOTEBOOKS=${ROOT_NOTEBOOKS:-"01_build_multimodal_features.ipynb 02_train_classifier_es.ipynb 03_train_classifier_en.ipynb 04_train_classifier_es_en.ipynb 05_exp_audio_vad.ipynb 06_exp_video_clean.ipynb 07_exp_sensorial_filter.ipynb 08_exp_fusion_avanzada.ipynb 09_robust_hpo_models.ipynb 10_stacking_ensemble_hpo.ipynb 11_boosting_multimodal_hpo.ipynb 12_deep_gated_multimodal_hpo.ipynb 13_meta_ensemble_weight_search.ipynb 14_video_only_siglip_xgb.ipynb 15_qwen_text_video_xgb.ipynb 16_qwen_text_video_sensor_xgb.ipynb"}

ENTREGABLES_DIR="entregables"
PREDICTION_DIR="prediction"

mkdir -p "$ENTREGABLES_DIR" "$PREDICTION_DIR"

declare -a NOTEBOOK_QUEUE=()

if [ "$INCLUDE_ROOT_NOTEBOOKS" = "1" ]; then
    for NOTEBOOK in $ROOT_NOTEBOOKS; do
        NOTEBOOK_QUEUE+=("$NOTEBOOK")
    done
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
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
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
    for json_file in "$ENTREGABLES_DIR"/*.json; do
        if [ -e "$json_file" ]; then
            cp -f "$json_file" "$PREDICTION_DIR"/
            copied_any=1
        fi
    done

    if [ "$copied_any" -eq 1 ]; then
        total_json=$(ls -1 "$PREDICTION_DIR"/*.json 2>/dev/null | wc -l)
        echo "JSON sincronizados en $PREDICTION_DIR (total: $total_json)"
    else
        echo "No hay JSON nuevos en $ENTREGABLES_DIR para sincronizar"
    fi
    
    echo ""
done

echo "=========================================="
echo "All notebooks processed!"
echo "Prediction dir: $PREDICTION_DIR"
echo "=========================================="