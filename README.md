This repository contains the code related to my MSc thesis: How Far Can a Few Shots Take? Exploring Few-Shot Learning in Finnish Text Classification Tasks Through Sentence Transformer Fine-Tuning.

## Data

Download and uncompress the data used in this study to replicate results to directory

```
./data/[TASK NAME]
```

Task names used in this repo are "fincore", "fincore-upper", "toxicity", "yle" and "ylilauta". 

FinCORE: https://github.com/TurkuNLP/FinCORE_full

Toxicity challenge: https://huggingface.co/datasets/TurkuNLP/jigsaw_toxicity_pred_fi

Yle corpus: https://github.com/spyysalo/yle-corpus

Ylilauta corpus: https://github.com/spyysalo/ylilauta-corpus

## Creating train datasets

Run sample_data.py to create 10 randomised datasets with n examples per label. For example, to create training sets for FinCORE with 8 examples per label, run

```
python sample_data.py --train ./data/fincore/train.tsv --task fincore --n_samples 8
```

## Training

The code includes scripts for running them in CSC puhti environment. 

To train a model, use slurm_train.sh script with the following arguments:

```
sbatch slurm_train.sh [TASK] [N_SAMPLES] [ITERATION NUMBER]
```

E.g., to run train.py for the previously created FinCORE datasets, run

```
sbatch slurm_train.sh slurm_train.sh fincore 8 "0 1 2 3 4 5 6 7 8 9"
```

Do note that you need to specify (or comment out) the desired model in the slurm_train.sh file.

## Evaluation

The classification report is saved in the logs. If you want to examine the predictions further, use a previously saved model to do predictions on the test set.

```
sbatch slurm_predict.sh [MODEL] [MODEL ALIAS] [TASK]
```

The models are saved in ./checkpoints/MODEL-TASK-N_SAMPLES-EPOCHS-ITERATION E.g., to run predictions on a model trained in the last step, run

```
sbatch slurm_predict.sh ./checkpoints/TurkuNLP/sbert-cased-finnish-paraphrase-fincore-8-1-0/ finsbert-fincore-8.0 fincore
```
