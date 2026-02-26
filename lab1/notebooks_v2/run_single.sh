#!/bin/bash
#SBATCH -p long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:1

# El nombre del notebook se pasa como argumento
NOTEBOOK=$1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate RFA2526pt

mkdir -p entregables

echo "=========================================="
echo "Starting execution of: $NOTEBOOK"
echo "=========================================="

python -c "
import os, nbformat
from nbconvert.preprocessors import ExecutePreprocessor

input_file = '$NOTEBOOK'
out_dir = 'entregables'

os.makedirs(out_dir, exist_ok=True)

# Prepare output filename
base_name = input_file.replace('.ipynb', '_out.ipynb')
output_file = os.path.join(out_dir, base_name)

counter = 1
root, ext = os.path.splitext(base_name)
while os.path.exists(output_file):
    output_file = os.path.join(out_dir, f'{root}_{counter}{ext}')
    counter += 1

# Prepare error filename
error_base_name = input_file.replace('.ipynb', '_error.ipynb')
error_output_file = os.path.join(out_dir, error_base_name)

error_counter = 1
error_root, error_ext = os.path.splitext(error_base_name)
while os.path.exists(error_output_file):
    error_output_file = os.path.join(out_dir, f'{error_root}_{error_counter}{error_ext}')
    error_counter += 1

with open(input_file) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')

try:
    ep.preprocess(nb, {'metadata': {'path': './'}})
    # Save successful execution
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Successfully saved to: {output_file}')
except Exception as e:
    # Save notebook with error state
    with open(error_output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Error occurred. Notebook saved to: {error_output_file}')
    print(f'Error: {e}')
    exit(1)
"

echo "Finished processing: $NOTEBOOK"
