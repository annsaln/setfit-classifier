# SetFit classifier

This repository contains the code related to my MSc thesis: How Far Can a Few Shots Take? Exploring Few-Shot Learning in Finnish Text Classification Tasks Through Sentence Transformer Fine-Tuning.

# Quickstart

The code includes scripts for running them in CSC Puhti environment. 

You can test out the code by using the emotion recognition dataset straight from the HuggingFace SetFit repo, by running 

```
python train.py --task test
```

You can also use the slurm script for batching jobs on Puhti:

```
sbatch slurm_train.sh test [N_SAMPLES] [ITERATION NUMBER]
```

e.g. 

```
sbatch slurm_train.sh test 8 "0 1 2"
```

This should approximately replicate the results in the original paper: https://arxiv.org/abs/2209.11055

# Data

Download and uncompress the data used in this study to replicate results to directory

```
./data/[TASK NAME]
```

Task names used in this repo are "fincore", "fincore-upper", "toxicity", "yle" and "ylilauta". 

Due to copyright reasons, I will not distribute the data used in my thesis in this repository. However, you can access the data used by following these links: 

FinCORE: https://github.com/TurkuNLP/FinCORE_full

Toxicity challenge: https://huggingface.co/datasets/TurkuNLP/jigsaw_toxicity_pred_fi

Yle corpus: https://github.com/spyysalo/yle-corpus

Ylilauta corpus: https://github.com/spyysalo/ylilauta-corpus

Please do note that the Yle and Ylilauta corpora are not distributed for open access. 

## Creating train datasets

Run sample_data.py to create 10 randomised datasets with n examples per label. For example, to create training sets for FinCORE with 8 examples per label, run

```
python sample_data.py --train ./data/fincore/train.tsv --task fincore --n_samples 8
```

# Fine-tuning with SetFit



To fine-tune a model, run

```
python train.py --model [MODEL] --datapath [PATH TO DATA DIR] --train [PATH TO TRAIN SET FILE] --task [TASK] --n_samples [NUMBER OF SAMPLES PER LABEL]
```

E.g. to run train.py for one of the previously created FinCORE datasets, run

```
python train.py --model "TurkuNLP/sbert-cased-finnish-paraphrase" --datapath ./data/fincore/ --train ./data/fincore/train.tsv.8.0 --task fincore --n_samples 8
```

You can also use slurm_train.sh script with the following arguments:

```
sbatch slurm_train.sh [TASK] [N_SAMPLES] [ITERATION NUMBER]
```

E.g.

```
sbatch slurm_train.sh fincore 8 "0 1 2 3 4 5 6 7 8 9"
```

Do note that you need to specify (or comment out) the desired model in the slurm_train.sh file, otherwise TurkuNLP's Finnish SBERT will be used as a default. 


# Evaluation

The classification report is saved in the logs directory when running with the slurm script. If you want to examine the predictions further, use a previously saved model to do predictions on the test set.

```
python predict_test.py --model [MODEL] --datapath [PATH TO DATA DIR] --task [TASK] --output_file [OUTPUT FILE NAME]
```

Or with the slurm script:

```
sbatch slurm_predict.sh [MODEL] [MODEL ALIAS] [TASK]
```

The models are saved in ./checkpoints/MODEL-TASK-N_SAMPLES-EPOCHS-ITERATION E.g., to run predictions on a model trained in the last step, run

```
sbatch slurm_predict.sh ./checkpoints/TurkuNLP/sbert-cased-finnish-paraphrase-fincore-8-1-0/ finsbert-fincore-8.0 fincore
```
