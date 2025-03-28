from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from csv import writer
from loaders import load_data
from random import randint
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import (
    classification_report,
)
import datasets
import sys


# default arguments
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
TRAIN_EPOCHS = 1
MODEL_NAME = "TurkuNLP/sbert-cased-finnish-paraphrase"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name", default=MODEL_NAME, help="Pretrained model name")
    ap.add_argument("--datapath")
    ap.add_argument("--train", default=None, help="Path to training data")
    ap.add_argument(
        "--batch_size",
        metavar="INT",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training",
    )
    ap.add_argument(
        "--epochs",
        metavar="INT",
        type=int,
        default=TRAIN_EPOCHS,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--learning_rate",
        metavar="FLOAT",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--sampling_strategy", type=str, default="unique")
    ap.add_argument(
        "--task",
        type=str,
        default="fincore",
        choices=["fincore", "fincore-upper", "toxicity", "yle", "ylilauta", "test"],
    )
    ap.add_argument("--output_dir", default="checkpoints")
    return ap


options = argparser().parse_args(sys.argv[1:])
learning_rate = options.learning_rate
print(options.train)

# You can run a test to replicate the results from the original Tunstall et al. paper
if options.task == "test":
    dataset = datasets.load_dataset("SetFit/emotion")
    dataset["train"] = sample_dataset(
        dataset["train"], label_column="label", num_samples=8, seed=randint(0, 100)
    )
    dataset["dev"] = dataset["validation"]
    labels = list(set(dataset["train"]["label_text"]))
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2", trust_remote_code=True
    )

# Otherwise, load data with a script
else:
    dataset, labels = load_data(options)
    # Load SetFit model from Hub
    if options.task in ["fincore", "fincore-upper", "toxicity"]:
        model = SetFitModel.from_pretrained(
            options.model_name,
            multi_target_strategy="one-vs-rest",
            trust_remote_code=True,
        )
    else:
        model = SetFitModel.from_pretrained(options.model_name, trust_remote_code=True)

# Custom evaluation metrics function for SetFit.trainer to get more fine-grained results
def eval_metrics(y_pred, y_test, labels=labels):
    print(classification_report(y_test, y_pred, target_names=labels))
    res = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    metrics = {"f1": res['micro avg']['f1-score'], "precision": res['micro avg']['precision'], "recall": res['micro avg']['recall']}
    return metrics

# Set epochs as 0 to run baseline experiments
if options.epochs == 0:
    args = TrainingArguments(
        batch_size=options.batch_size,
        num_epochs=0,  # run only classification head training with default number of epochs
        warmup_proportion=0.0,
        seed=randint(0,100),
        output_dir=f'{options.output_dir}-BL'
    )        
else:
    args = TrainingArguments(
        batch_size=options.batch_size,
        num_epochs=(
            options.epochs,
            16,
        ),  # set number of epochs for ST embedding and classification head training
        body_learning_rate=(options.learning_rate, 1e-5),
        loss=CosineSimilarityLoss,
        sampling_strategy=options.sampling_strategy,
        logging_steps=500,
        output_dir=options.output_dir,
    )


# Create trainer
trainer = Trainer(
    args=args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    metric=eval_metrics,
)

# Train the model
trainer.train()

# Evaluate the model, print the results in a file
metrics = trainer.evaluate(dataset["test"])
print(metrics)
i = options.train[-1]
with open(f"output/{options.task}-stats.tsv", "a") as f:
    csv_writer = writer(f, delimiter="\t", lineterminator="\n")
    row = [
        options.model_name,
        options.epochs,
        options.n_samples,
        i,
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    ]
    if f.tell() == 0:
        csv_writer.writerows(
            [
                [
                    "model",
                    "epochs",
                    "samples",
                    "iteration",
                    "f1",
                    "precision",
                    "recall",
                ],
                row,
            ]
        )
    else:
        csv_writer.writerow(row)
    f.close()

# save trained model
trainer.model.save_pretrained(options.output_dir)