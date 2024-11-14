from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train", required=True, help="Path to training data")
    ap.add_argument(
        "--task",
        default="fincore",
        choices=["fincore", "fincore-upper", "toxicity", "yle", "ylilauta"],
    )
    ap.add_argument("--n_samples", type=int, default=8)
    return ap


options = argparser().parse_args(sys.argv[1:])
seed = np.random.RandomState(42)


# the data should have n samples per class, not n samples per class combination
# all the sub-register examples have two labels: main register and subregister
# because of this, main register labels will appear at least n * subregister times
# include only examples from subregisters that are not hybrids in the training set?
# TODO: fix this

if "fincore" in options.task:
    format = "csv"
    labels_file = open("./data/fincore/fincore_labels.txt", "r")
    if options.task == "fincore":
        labels = [l.split("=")[0].strip() for l in labels_file.read().splitlines()]
    else:
        labels = [
            l.split("=")[0].strip()
            for l in labels_file.read().splitlines()
            if "MAIN" in l
        ]
    data = pd.read_csv(
        options.train, delimiter="\t", names=["label", "text"], keep_default_na=False
    )

if options.task == "toxicity":
    format = "jsonl"
    df = pd.read_json(options.train, lines=True)
    data, test_set = train_test_split(df, test_size=0.2, random_state=42)
    labels = data.columns[1:-2].tolist()
    data = data.reset_index(drop=True)
    data["label_clean"] = [1 if sum(data.iloc[i, 1:-2]) == 0 else 0 for i in data.index]
    labels.append("label_clean")
    test_set.to_json(f"data/toxicity/dev_fi_deepl.jsonl", orient="records", lines=True)

if options.task in ["ylilauta", "yle"]:
    format = "txt"
    f = open(options.train, "r")
    content = f.read().splitlines()
    linelabels = [line.split()[0] for line in content]
    texts = [" ".join(line.split()[1:]) for line in content]
    f.close()
    data = pd.DataFrame(list(zip(linelabels, texts)), columns=["label", "text"])
    labels = data.label.unique().tolist()

for i in range(10):
    label_dfs = []
    for label in labels:
        if options.task in ["fincore", "fincore-upper", "yle", "ylilauta"]:
            label_df = data[data["label"].str.contains(label)]
        elif options.task == "toxicity":
            label_df = data[data[label] == 1]
        if label_df.shape[0] >= options.n_samples:
            sample_df = label_df.sample(n=options.n_samples, random_state=i)
            label_dfs.append(sample_df)
        else:
            label_dfs.append(label_df)
    train_examples = pd.concat(label_dfs)
    if format == "csv":
        train_examples.to_csv(
            f"{options.train}.{options.n_samples}.{i}",
            sep="\t",
            index=False,
            header=False,
        )
    elif format == "jsonl":
        train_examples = train_examples.drop("label_clean", axis=1)
        train_examples.to_json(
            f"{options.train}.{options.n_samples}.{i}", orient="records", lines=True
        )
    elif format == "txt":
        train_examples.to_csv(
            f"{options.train}.{options.n_samples}.{i}",
            sep=" ",
            index=False,
            header=False,
        )
