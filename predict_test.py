from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from loaders import load_data
from setfit import SetFitModel
import sys
import pandas as pd
import numpy as np


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", help="Pretrained model path")
    ap.add_argument("--datapath")
    ap.add_argument(
        "--task",
        type=str,
        default="fincore",
        choices=["fincore", "fincore-upper", "toxicity", "yle", "ylilauta", "test"],
    )
    ap.add_argument("--output_file", default="output.json")
    return ap


options = argparser().parse_args(sys.argv[1:])

# Load data with a script
dataset, labels = load_data(options)
dataset = dataset["test"]
print("data loaded")

# Load pre-trained model for validation
model = SetFitModel.from_pretrained(options.model)
# Extract labels from model
labels = model.id2label
print("model loaded")

# Predict label probabilities for each example
preds = np.asarray(
    model.predict_proba(dataset["text"], batch_size=8, show_progress_bar=True)
)

# Extract embeddings from the fine-tuned embedding model
embeddings = model.encode(
    dataset["text"], batch_size=8, show_progress_bar=True
).tolist()  # embeddings

df = pd.DataFrame(
    list(zip(dataset["text"], dataset["label"], preds)),
    columns=["text", "true", "preds"],
)
for i in range(len(labels)):
    df[labels[i]] = preds[:, i]
df["embedding"] = embeddings
df.to_json(options.output_file, index=False) # save to file
