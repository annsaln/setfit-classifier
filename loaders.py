import datasets
import pandas as pd
from sentence_transformers.losses import CosineSimilarityLoss
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)
import numpy as np
from collections import Counter
from csv import writer


def load_fincore(args):
    labels_file = open("./data/fincore/fincore_labels.txt", "r")
    if args.task == "fincore":
        labels = [l.split("=")[0].strip() for l in labels_file.read().splitlines()]
    else:
        # ID doesn't have "MAIN" tag even though it is a main register
        labels = [
            l.split("=")[0].strip()
            for l in labels_file.read().splitlines()
            if "MAIN" in l or "ID" in l
        ]

    if "train" in args:
        datafiles = {
            "dev": f"{args.datapath}/dev.tsv",
            "test": f"{args.datapath}/test.tsv",
            f"train": args.train,
        }
    else:
        # load only test set for prediction
        datafiles = {"test": f"{args.datapath}/test.tsv"}

    dataset = datasets.load_dataset(
        "csv",
        data_files=datafiles,
        delimiter="\t",
        column_names=["label", "text"],
        keep_default_na=False,
        cache_dir="cachedir",
    )

    dataset = dataset.map(lambda line: {"text": line["text"].strip('"')})
    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit(labels)
    print("Binarizing the labels")
    dataset = dataset.map(lambda example: {"label": example["label"].split(" ")})
    # check that there are no extra labels that do not belong to the dataset
    dataset = dataset.map(
        lambda example: {"label": [l for l in example["label"] if l in labels]}
    )
    dataset = dataset.map(
        lambda example: {"label": mlb.transform([example["label"]])[0]}
    )
    return dataset, labels


def load_toxicity(args):
    if "train" in args:
        datafiles = {
            "dev": f"{args.datapath}/dev_fi_deepl.jsonl",
            "test": f"{args.datapath}/test_fi_deepl.jsonl",
            "train": args.train,
        }
    else:
        datafiles = {"test": f"{args.datapath}/test_fi_deepl.jsonl"}
    dataset = datasets.load_dataset("json", data_files=datafiles, cache_dir="cachedir")
    labels = [c for c in dataset["test"].column_names if "label" in c]
    dataset = dataset.map(
        lambda example: {
            "label_clean": 0 if sum([example[label] for label in labels]) > 0 else 1
        }
    )
    labels.append("label_clean")
    dataset = dataset.map(
        lambda example: {"label": [example[label] for label in labels]}
    )
    return dataset, labels


def load_txt(args):
    if "train" in args:
        datafiles = {
            "dev": f"{args.datapath}/dev.txt",
            "test": f"{args.datapath}/test.txt",
            "train": args.train,
        }
    else:
        datafiles = {"test": f"{args.datapath}/test.txt"}
    dataset = datasets.load_dataset("text", data_files=datafiles, cache_dir="cachedir")
    dataset = dataset.map(lambda example: {"label": example["text"].split()[0]})
    dataset = dataset.map(
        lambda example: {"text": " ".join(example["text"].split()[1:]).strip('"')}
    )
    labels = sorted(list(set(dataset["test"]["label"])))
    return dataset, labels


def load_data(args: dict):
    if "fincore" in args.task:
        return load_fincore(args)
    elif args.task == "toxicity":
        return load_toxicity(args)
    elif args.task in ["yle", "ylilauta"]:
        return load_txt(args)
