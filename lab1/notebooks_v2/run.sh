#!/bin/bash
#SBATCH -p long
#SBATCH --cpus-per-task=8
#SBATCH --job-name=run
#SBATCH --mem=32G
#SBATCH --gres=shard:4
#SBATCH -o logs/%j.log

# Lista de notebooks a ejecutar (separados por espacios)
NOTEBOOKS="03_f2llm_4B_ft_tweet.ipynb 03_modelos_clasicos.ipynb 04_f2llm_4B_ft_text_clean.ipynb 05_KaLM_ft_tweet.ipynb 06_KaLM_ft_text_clean.ipynb 07_Ministral3_8B_inference_tweet.ipynb 08_Ministral3_8B_only_ft.ipynb 09_Ministral3_8B_inference_ft.ipynb"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate RFA2526pt

# Configure CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ejecutar cada notebook secuencialmente
for NOTEBOOK in $NOTEBOOKS; do
    echo "=========================================="
    echo "Starting execution of: $NOTEBOOK"
    echo "=========================================="
    
    export NB_IN="$NOTEBOOK"
    
    python << 'EOF'
import os, nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import torch
import gc

input_file = os.environ['NB_IN']
out_dir = "entregables"

os.makedirs(out_dir, exist_ok=True)

# Prepare output filename
base_name = input_file.replace('.ipynb', '_out.ipynb')
output_file = os.path.join(out_dir, base_name)

counter = 1
root, ext = os.path.splitext(base_name)
while os.path.exists(output_file):
    output_file = os.path.join(out_dir, f"{root}_{counter}{ext}")
    counter += 1

# Prepare error filename
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
    fi
    
    echo ""
done

echo "=========================================="
echo "All notebooks processed!"
echo "=========================================="