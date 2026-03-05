#!/bin/bash
#SBATCH -p long
#SBATCH --cpus-per-task=8
#SBATCH --job-name=vision
#SBATCH --mem=32G
#SBATCH --gres=shard:4
#SBATCH --array=0-3
#SBATCH -o logs/vision_%A_%a.log

NOTEBOOKS=(
    "10_internvl25_binary_classification.ipynb"
    "11_qwen25vl_binary_classification.ipynb"
    "13_pixtral12b_binary_classification.ipynb"
)

NOTEBOOK="${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate RFA2526pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Starting: $NOTEBOOK"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

python << PYEOF
import os, nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import torch
import gc

input_file = "$NOTEBOOK"
out_dir = "entregables"
os.makedirs(out_dir, exist_ok=True)

base_name = input_file.replace('.ipynb', '_out.ipynb')
output_file = os.path.join(out_dir, base_name)

counter = 1
root, ext = os.path.splitext(base_name)
while os.path.exists(output_file):
    output_file = os.path.join(out_dir, f"{root}_{counter}{ext}")
    counter += 1

error_base_name = input_file.replace('.ipynb', '_error.ipynb')
error_output_file = os.path.join(out_dir, error_base_name)

error_counter = 1
error_root, error_ext = os.path.splitext(error_base_name)
while os.path.exists(error_output_file):
    error_output_file = os.path.join(out_dir, f"{error_root}_{error_counter}{error_ext}")
    error_counter += 1

with open(input_file) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')

try:
    ep.preprocess(nb, {'metadata': {'path': './'}})
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully saved to: {output_file}")
except Exception as e:
    with open(error_output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Error. Saved to: {error_output_file}")
    print(f"Error: {e}")
finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
PYEOF

echo "Finished: $NOTEBOOK"
