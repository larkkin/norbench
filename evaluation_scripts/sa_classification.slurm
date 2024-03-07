#!/bin/bash
#SBATCH --job-name=norbench
#SBATCH --account=nn9851k
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=accel

# Load modules
module --quiet purge
module load nlpl-scipy-ecosystem/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-nlptools/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-python-candy/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-pytorch/1.11.0-gomkl-2021a-cuda-11.3.1-Python-3.9.5
module load nlpl-transformers/4.24.0-gomkl-2021a-Python-3.9.5

export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

export PYTHONUSERBASE='/projappl/project_465000498/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH


MODEL=${1}  # Path to the model or its HuggingFace name
IDENTIFIER=${2}  # Identifier to save the results
TYPE=${3}  # document or sentence? (type of the task)

echo ${MODEL}
echo ${IDENTIFIER}
echo ${TYPE}

for SEED in 10 20 30 40 50
do
  echo ${SEED}
  python3 sa_classification.py  -m ${MODEL} -i ${IDENTIFIER} --type ${TYPE} -d ../sentiment_analysis/sentence_mixed_class/${TYPE}/train.csv -dev ../sentiment_analysis/sentence_mixed_class/${TYPE}/dev.csv -t ../sentiment_analysis/sentence_mixed_class/${TYPE}/test.csv -s ${SEED}
done

python3 analysis.py scores/${IDENTIFIER}_${TYPE}.tsv
echo scores/${IDENTIFIER}_${TYPE}.tsv
