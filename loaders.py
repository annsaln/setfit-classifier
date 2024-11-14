import datasets
import pandas as pd
from sentence_transformers.losses import CosineSimilarityLoss
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import numpy as np
from collections import Counter
from csv import writer



def load_fincore(args):
    labels_file = open("./data/fincore/fincore_labels.txt", "r")
    if args.task == 'fincore':
        labels = [l.split("=")[0].strip() for l in labels_file.read().splitlines()]
    else:
        labels = [l.split("=")[0].strip() for l in labels_file.read().splitlines() if 'MAIN' in l]

    datafiles={'dev': f"{args.datapath}/dev.tsv", "test": f"{args.datapath}/test.tsv", f"train": args.train}
#    if args.train = None:
#        datafiles['train0'] = args.train
#    for i in range(args.n_iterations):
#        datafiles[f'train{i}'] = f"{args.datapath}/train.tsv.{args.n_samples}.{i}"
    dataset = datasets.load_dataset(
        "csv",
        data_files=datafiles,
        delimiter="\t",
        column_names=['label', 'text'],
        keep_default_na=False,
        cache_dir="cachedir"
        )

    dataset = dataset.map(lambda line: {'text': line['text'].strip('"')})
    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit(labels)
    print("Binarizing the labels")
    dataset = dataset.map(lambda example: {'label': example['label'].split(' ')})
    # check that there are no extra labels that do not belong to the dataset
    dataset = dataset.map(lambda example: {'label': [l for l in example['label'] if l in labels]})
#    dataset = dataset.filter(lambda example: set(example['label']).issubset(labels))
    dataset = dataset.map(lambda example: {'label': mlb.transform([example['label']])[0]})
    return dataset, labels


def load_toxicity(args):
    datafiles={'dev': f"{args.datapath}/dev_fi_deepl.jsonl", "test": f"{args.datapath}/test_fi_deepl.jsonl", "train": args.train}
#    for i in range(args.n_iterations):
#        datafiles[f'train{i}'] = f"{args.datapath}/train_fi_deepl.jsonl.{args.n_samples}.{i}"
    dataset = datasets.load_dataset("json", data_files=datafiles, cache_dir="cachedir") # you should just split the train file into dev and train --> this should've been done already 
    labels = [c for c in dataset['train'].column_names if 'label' in c]
    dataset = dataset.map(lambda example: {'label_clean': 0 if sum([example[label] for label in labels])>0 else 1})
    labels.append('label_clean')
    dataset = dataset.map(lambda example: {"label": [example[label] for label in labels]})
    return dataset, labels

def load_txt(args):
    datafiles={'dev': f"{args.datapath}/dev.txt", "test": f"{args.datapath}/test.txt", "train": args.train}
    dataset = datasets.load_dataset("text", data_files=datafiles, cache_dir="cachedir")
    dataset = dataset.map(lambda example: {"label": example["text"].split()[0]})
    dataset = dataset.map(lambda example: {"text": " ".join(example["text"].split()[1:]).strip('"')})
    labels = list(set(dataset["train"]["label"]))
    return dataset, labels


def load_data(args:dict):
    if 'fincore' in args.task:
        return load_fincore(args)
    elif args.task == 'toxicity':
        return load_toxicity(args)
    elif args.task in ["yle", "ylilauta"]:
        return load_txt(args)
    