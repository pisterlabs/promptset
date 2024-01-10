from argparse import ArgumentParser
from transformers import (
    Trainer,
    AutoConfig,
    AutoModelWithHeads,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    XLMRobertaForMaskedLM,
)
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import logging
import torch
import pandas as pd
import json

from DialEvalML.utils.utils import *
from DialEvalML.mlm import MLM_predictor

import openai

openai.api_key = "key"
MAX_LENGTH = 512
TURN_LVL_QUALITIES = [
    "APPROPRIATENESS",
    "CONTENT_RICHNESS",
    "GRAMMATICAL_CORRECTNESS",
    "RELEVANCE",
]

models = {
    "nsp": {
        "nsp_paL_siamese": "exp/xlm-roberta-large/nsp/paL_siamese",
        "nsp_paL_concat": "exp/xlm-roberta-large/nsp/paL",
        "nsp_ml75": "exp/xlm-roberta-large/nsp/ml75",
    },
    "vsp": {
        "vsp_ml5": "exp/xlm-roberta-large/vsp/ml5",
        "vsp_en": "exp/xlm-roberta-large/vsp/en",
        "vsp_paL": "exp/xlm-roberta-large/vsp/paL",
    },
    "engagement": {
        "endex_ml10": "exp/xlm-roberta-large/eng/endex_ml10",
        "endex_ml20": "exp/xlm-roberta-large/eng/endex_ml20",
        "endex_ml50": "exp/xlm-roberta-large/eng/endex_ml50",
    },
}

gpt_metrics = {
    "gpt-overall": (
        "Given the Context, evaluate from 1-5 the Response in terms of Appropriateness. Provide a single score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Overall Quality. Provide a single score and nothing else.",
    ),
    "gpt-relevant": (
        "Given the Context, evaluate from 1-5 the Response in terms of Relevance. Provide a single score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Coherence. Provide a single score and nothing else.",
    ),
    "gpt-engaging": (
        "Given the Context, evaluate from 1-5 the Response in terms of Content Richness. Provide a single score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Likeability. Provide a single score and nothing else.",
    ),
    "gpt-content": (
        "Evaluate from 1-5 the Response in terms of Grammatical Correctness. Provide a single score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Informativeness. Provide a single score and nothing else.",
    ),
}



def preprocess_ctxres(examples):
    args = (examples["ctx"], examples["res"])
    result = tokenizer(
        *args, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )
    return result


def preprocess_res(examples):
    args = (examples["res"],)
    result = tokenizer(
        *args, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )
    return result


preprocess_funcs = {
    "vsp": preprocess_res,
    "nsp": preprocess_ctxres,
    "engagement": preprocess_ctxres,
}

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = ArgumentParser(
        prog="DialEvalML predictor",
        description="Evalutes a response given the preceeding context.",
    )
    parser.add_argument("--dev_dir", default= "DSTC11/dev",help="development folder")
    parser.add_argument("--data_path", help= "path to preprocessed data to evaluate")
    parser.add_argument("--weights", help="Weights .json", default="DSTC11/weights/weights_all_crs.json")
    parser.add_argument("--out", help="output file name", default="results")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)

    args = parser.parse_args()

    submetrics = [
        "vsp",
        "nsp",
        "mlm",
        "engagement",
        "gpt-overall",
        "gpt-relevant",
        "gpt-engaging",
        "gpt-content",
    ]

    training_args = TrainingArguments(
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_predict=True,
        output_dir="exp/out",
        label_names=["labels"],
    )
    config = AutoConfig.from_pretrained(
        "xlm-roberta-large",
        num_labels=1,
    )
    special_token_dict = {
        "speaker1_token": "<speaker1>",
        "speaker2_token": "<speaker2>",
    }

    tokenizer = AutoTokenizer.from_pretrained(
        "xlm-roberta-large",
        use_fast=True,
        truncation_side="left",
    )
    tokenizer.add_tokens(list(special_token_dict.values()))

    predictions = dict()
    gpt_predictions_turn = dict()
    gpt_predictions_dial = dict()
    for submetric in submetrics:
        predictions[submetric] = dict()
        if submetric in ["nsp", "vsp", "engagement"]:
            for model_name, model_path in models[submetric].items():
                predictions[submetric][model_name] = dict()
                model = AutoModelWithHeads.from_pretrained(
                    models[submetric][model_name],
                    config=models[submetric][model_name],
                )
                model.resize_token_embeddings(len(tokenizer))
                model.to(device)

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=None,
                    eval_dataset=None,
                    tokenizer=tokenizer,
                    data_collator=default_data_collator,
                )
                ds = load_dataset(
                    "csv",
                    data_files={"test": args.data_path},
                    download_mode="force_redownload",
                )
                ds = ds.map(preprocess_funcs[submetric], batched=True)
                preds, _, metadata = trainer.predict(test_dataset=ds["test"])
                predictions[submetric][model_name] = preds.tolist()

        elif submetric == "mlm":
            model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large")
            model.to(device)

            mlm_predictor = MLM_predictor(model, tokenizer, device)

            data_test = pd.read_csv(args.data_path)
            predictions[submetric] = [
                mlm_predictor.predict(x[10:]) for x in tqdm(data_test.res)
            ]

        elif submetric in [
            "gpt-overall",
            "gpt-relevant",
            "gpt-engaging",
            "gpt-content",
        ]:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            prompt = gpt_metrics[submetric][0]
            data_test = load_dataset(
                    "csv",
                    data_files={"test": args.data_path},
                    download_mode="force_redownload",
                )["test"]
            preds = []
            for i in tqdm(range(0, len(data_test))):
                ctx = data_test[i]["ctx"]
                res = data_test[i]["res"]
                text_prompt = f"Context:{ctx}\nResponse:{res}"
                j = 1
                # prevent context overflow. start removing from older context
                while len(tokenizer(text_prompt)[0]) > 3900:
                    new_ctx = "\n".join(data_test[i]["context"].splitlines()[j:])
                    text_prompt = f"Context:{new_ctx}\nResponse:{res}"
                    j += 1

                gpt_score = -1
                score_backoff = 0
                while gpt_score == -1 and score_backoff < 2:
                    completion = completions_with_backoff(
                        model="gpt-3.5-turbo",
                        temperature=0.0,
                        max_tokens=20,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": text_prompt},
                        ],
                    )
                    gpt_score = completion_to_score(
                        completion.choices[0].message.content
                    )
                    score_backoff += 1
                if gpt_score != -1:
                    preds.append(gpt_score)
                else:
                    preds.append(3)

            gpt_predictions_turn[submetric] = preds


    with open(args.weights, "r") as outfile:
        weights = json.load(outfile)

    turn_weights = dict(
        [(key, value) for key, value in weights.items() if key in TURN_LVL_QUALITIES]
    )

    turn_preds = {**predictions, **gpt_predictions_turn}

    test_normalize(turn_preds, args.dev_dir)

    final_preds = list()
    fused_prediction = dict()

    for submetric in submetrics:
        if submetric in ["nsp", "vsp", "engagement"]:
            avg = []
            for model in turn_preds[submetric].keys():
                avg.append(turn_preds[submetric][model])
            final_preds.append(np.mean((avg), axis=0))
        else:
            final_preds.append(turn_preds[submetric])
    data = {}
    for quality in turn_weights:
        data[quality] = np.dot(turn_weights[quality], (final_preds))

    data.to_csv(f"{args.out}.csv", index=False)
