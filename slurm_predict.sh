#!/bin/bash
#SBATCH --account=Project_2002026         # Billing project, has to be defined!
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --gres=gpu:v100:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
##SBATCH --mail-type=BEGIN          # Uncomment to enable mail

echo "START: $(date)"

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load pytorch 

pip install -q --user datasets
pip install -q --user setfit
pip install -q --user huggingface_hub
pip install -q --user diffusers>=0.29.0

#MODEL="TurkuNLP/sbert-cased-finnish-paraphrase"
#MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" #mteb 148
#MODEL="intfloat/multilingual-e5-small"
#MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#MODEL="sentence-transformers/paraphrase-xlm-r-multilingual-v1"

MODEL=$1
MODEL_ALIAS=$2
TASK=$3

BS=8

echo "MODEL:$MODEL"
echo "TASK:$TASK"


export DATA_DIR=data/$TASK
export OUTPUT_DIR=preds/$TASK
export HF_HOME=cachedir

mkdir -p "$OUTPUT_DIR"
echo "Settings: model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS n_samples=$N_SAMPLES"
echo "job=$SLURM_JOBID  model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS n_samples=$N_SAMPLES" >> logs/experiments.log


srun python predict_test.py \
  --model $MODEL \
  --datapath $DATA_DIR \
  --task $TASK \
  --output_file $OUTPUT_DIR/$MODEL_ALIAS.json


echo "job=$SLURM_JOBID task=$TASK model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS n_samples=$N_SAMPLES" >> logs/completed.log
seff $SLURM_JOBID
echo "END: $(date)"
