from datasets.version import DATA_VERSION
import openai
from utils import load_wandb_file
import numpy as np
import pandas as pd
import time
import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_gpt3_train_data(run, save_dir):
    preds = np.load(load_wandb_file("train_ensemble_preds.npy", run, save_dir))
    correct_preds = np.load(
        load_wandb_file("train_ensemble_correct_preds.npy", run, save_dir)
    )

    uncertain_keep = n_most_uncertain(preds, correct_preds, 5000)

    random_train = np.random.choice(np.where(~uncertain_keep)[0], 5000, replace=False)
    train = np.append(random_train, np.where(uncertain_keep))

    filename = "../twitter-datasets/full_train_ensemble" + "_v{}.csv".format(
        DATA_VERSION
    )
    df = pd.read_csv(filename)
    tweets = df.iloc[train]
    tweets = tweets.rename(columns={"texts": "prompt", "labels": "completion"})

    promptt = "{}\n Sentiment: "
    labels = {1: "positive", 0: "negative"}
    tweets["prompt"] = tweets["prompt"].apply(lambda x: promptt.format(x))
    tweets["completion"] = tweets["completion"].apply(lambda x: labels[x])
    tweets.to_csv("openai-parsed.csv", index=False)


def load_data():
    def read_txt(filename):
        with open(filename) as f:
            data = f.read().split("\n")
            data = [x for x in data if x != ""]
        # drop indices
        return [",".join(x.split(",")[1:]) for x in data]

    return read_txt("../twitter-datasets/test_data.txt")


def get_gpt3_predictions(run, args):
    test = np.load(load_wandb_file("test_preds.npy", run, args.save_dir))
    test_preds = test > 0.5
    df = pd.DataFrame(load_data(), columns=["texts"])
    masks = np.load(f"./masks/{args.mask}.pkl", allow_pickle=True)
    test_mask = masks[0]
    to_pred = df[test_mask]
    promptt = "{}\n Sentiment: "
    tweets = to_pred.texts.apply(lambda x: promptt.format(x))
    openai.api_key = "ADD_API_KEY"

    preds = [False] * len(tweets)
    for j, i in enumerate(tweets):
        w = openai.Completion.create(
            model=args.model, prompt=i, temperature=0, max_tokens=1
        )
        preds[j] = w["choices"][0]["text"] == " positive"
        time.sleep(3)

    test_preds[tweets.index] = preds
    final = (
        pd.DataFrame(test_preds, columns=["Prediction"])
        .reset_index(inplace=False)
        .rename(columns={"index": "Id"}, inplace=False)
    )
    final.Prediction = final.Prediction.replace({True: 1, False: 0})
    final.to_csv(f"test{run}.csv", index=False)


def uncertain(preds, thresh):
    return np.abs(preds - 0.5) <= thresh


def n_most_uncertain(preds, correct_preds, n):
    indexes = (np.abs(preds - 0.5) + correct_preds).argsort()[:n]
    final = np.full_like(preds, False, dtype=bool)
    final[indexes] = True
    return final


def get_n_mis_pred_uncertain(preds, labels, thresh):
    unc = uncertain(preds, thresh)
    mis_preds = (preds > 0.5) != labels
    return np.sum(mis_preds & unc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--mask", type=str)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join("/cluster/scratch", os.environ["USER"]),
    )
    parser.add_argument("--predict", type=str2bool, default=False)
    parser.add_argument(
        "--model", type=str, default="curie:ft-personal-2022-07-27-15-32-23"
    )

    args = parser.parse_args()
    if not args.predict:
        save_gpt3_train_data(args.run_id, args.save_dir)
    else:
        get_gpt3_predictions(args.run_id, args)
