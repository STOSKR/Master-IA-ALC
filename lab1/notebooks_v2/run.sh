#!/bin/bash

# Script launcher para ejecutar notebooks en paralelo
# Cada notebook se ejecuta como un job separado de SLURM

# Crear directorios necesarios si no existen
mkdir -p logs
mkdir -p entregables

echo "=========================================="
echo "Launching parallel jobs for all notebooks"
echo "=========================================="

# Función para enviar un job CON GPU
submit_job_gpu() {
    local notebook=$1
    local shards=$2
    local mem=$3
    local jobname=$4
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p long
#SBATCH --cpus-per-task=8
#SBATCH --job-name=${jobname}
#SBATCH --mem=${mem}
#SBATCH --gres=shard:${shards}
#SBATCH -o logs/${jobname}_%j.log

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate RFA2526pt

echo "=========================================="
echo "Starting execution of: ${notebook}"
echo "=========================================="

export NB_IN="${notebook}"

python << 'EOFPYTHON'
import os, nbformat
from nbconvert.preprocessors import ExecutePreprocessor

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
except Exception as e:
    # Save notebook with error state
    with open(error_output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Error occurred. Notebook saved to: {error_output_file}")
    print(f"Error: {e}")
    exit(1)
EOFPYTHON

echo "=========================================="
echo "Finished: ${notebook}"
echo "=========================================="
EOT
}

# Función para enviar un job SIN GPU (solo CPU)
submit_job_cpu() {
    local notebook=$1
    local mem=$2
    local jobname=$3
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p long
#SBATCH --cpus-per-task=8
#SBATCH --job-name=${jobname}
#SBATCH --mem=${mem}
#SBATCH -o logs/${jobname}_%j.log

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate RFA2526pt

echo "=========================================="
echo "Starting execution of: ${notebook}"
echo "=========================================="

export NB_IN="${notebook}"

python << 'EOFPYTHON'
import os, nbformat
from nbconvert.preprocessors import ExecutePreprocessor

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
except Exception as e:
    # Save notebook with error state
    with open(error_output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Error occurred. Notebook saved to: {error_output_file}")
    print(f"Error: {e}")
    exit(1)
EOFPYTHON

echo "=========================================="
echo "Finished: ${notebook}"
echo "=========================================="
EOT
}

# Lanzar jobs con recursos apropiados según el modelo

# Modelos clásicos - no necesitan GPU (solo CPU)
submit_job_cpu "03_modelos_clasicos.ipynb" "32G" "clasicos"

# F2LLM 4B - modelos de 4B parámetros (8GB VRAM)
submit_job_gpu "03_f2llm_4B_ft_tweet.ipynb" 1 "32G" "f2llm_tweet"
submit_job_gpu "04_f2llm_4B_ft_text_clean.ipynb" 1 "32G" "f2llm_clean"

# KaLM - similar a F2LLM (8GB VRAM)
submit_job_gpu "05_KaLM_ft_tweet.ipynb" 1 "32G" "kalm_tweet"
submit_job_gpu "06_KaLM_ft_text_clean.ipynb" 1 "32G" "kalm_clean"

# Ministral 8B - modelos más grandes (16GB VRAM)
submit_job_gpu "07_Ministral3_8B_inference_tweet.ipynb" 2 "48G" "ministral_inf"
submit_job_gpu "08_Ministral3_8B_only_ft.ipynb" 2 "48G" "ministral_ft"
submit_job_gpu "09_Ministral3_8B_inference_ft.ipynb" 2 "48G" "ministral_inf_ft"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Check 'squeue -u \$USER' to see job status"
echo "Logs will be in logs/ directory"
echo "=========================================="