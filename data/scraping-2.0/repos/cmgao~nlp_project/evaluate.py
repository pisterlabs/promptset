import argparse
import pandas as pd
from typing import Sequence, Callable
from sacrebleu.metrics import BLEU, CHRF
from openai import OpenAI
from opencc import OpenCC


def evaluate(ref: Sequence[str], pred: Sequence[str]) -> dict:
    """
    Evaluate the predictions against the reference using BLEU and chrF++.
    :param ref: list of reference sentences
    :param pred: list of predicted sentences
    :return: dictionary of scores
    """
    print("Reference:")
    print(ref[:5])
    print("Prediction:")
    print(pred[:5])

    bleu = BLEU(tokenize="zh")
    chrf = CHRF(word_order=2)

    bleu_score = bleu.corpus_score(pred, [ref]).score
    chrf_score = chrf.corpus_score(pred, [ref]).score

    return {"bleu": bleu_score, "chrf++": chrf_score}


def dummy_pred(ref: pd.DataFrame, target: str) -> Sequence[str]:
    """
    Use source text as naive baseline.
    """
    return ref.iloc[:, 1].tolist() if target == "yue" else ref.iloc[:, 0].tolist()


def pred_gpt(src: Sequence[str], target: str) -> Sequence[str | None]:
    """
    Use GPT-3.5 as prediction.
    """

    client = OpenAI()
    res = []
    cc = OpenCC('s2hk')

    prompt = {
        "cmn": "將下文從廣東話翻譯成中文普通話\n廣東話：",
        "yue": "將下文從中文普通話翻譯成廣東話\n普通話："
    }

    suffix = {
        "cmn": "\n普通話：",
        "yue": "\n廣東話："
    }

    for s in src:
        user_msg = prompt[target] + s + suffix[target]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You are helping in translations between Cantonese and Mandarin."},
                {"role": "user", "content": user_msg}
            ]
        )
        res_content = cc.convert(response.choices[0].message.content)
        print(res_content)
        res.append(res_content)

    return res


def evaluation_driver(input: str | Callable | None, split="val", domain="main", target="yue"):
    """
    :param input: If None, use source text as prediction.
                If callable, use it to generate predictions. It should take a list of source text and target language specifier.
                Otherwise, read from file.
    :param split: test or val
    :param domain: main or tatoeba
    :param target: yue or cmn
    """
    if split == "test":
        if domain == "main":
            ref = pd.read_csv("data/test_cleaned_parallel_sentences.txt", header=None).iloc[:, -2:]
        else:
            ref = pd.read_csv("data/test_tatoeba_sentences.txt", header=None)
    elif domain == "main":
        ref = pd.read_csv("data/valid_cleaned_parallel_sentences.txt", header=None).iloc[:, -2:]
    else:
        ref = pd.read_csv("data/valid_tatoeba_sentences.txt", header=None)

    if input is None:
        pred = dummy_pred(ref, target)
    elif callable(input):
        pred_input = dummy_pred(ref, target)
        pred = input(pred_input, target)
    else:
        with open(input, "r") as f:
            pred = f.readlines()

    ref_list = ref.iloc[:, 0].tolist() if target == "yue" else ref.iloc[:, 1].tolist()

    scores = evaluate(ref_list, pred)

    print(scores)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='split', action='store_const',
                    const="test", default="val",
                    help='Evaluate on test set instead of validation test')
    parser.add_argument('--input', help='Path to input file (contains predictions)', default=None)
    parser.add_argument("--domain", help='which data source to use (main or tatoeba)', choices=["main", "tatoeba"], required=True)
    parser.add_argument('--target', help='Target language (yue or cmn)', choices=["yue", "cmn"], required=True)
    parser.add_argument("--gpt", help="Use GPT-3.5 as prediction", action="store_true")

    args = parser.parse_args()

    input = pred_gpt if args.gpt else args.input
    evaluation_driver(input, args.split, args.domain, args.target)