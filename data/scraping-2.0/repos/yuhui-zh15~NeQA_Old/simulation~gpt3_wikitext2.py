# Evaluate GPT-3 on the WikiText-2 dataset.

from typing import List, Tuple

import random
import json
import datasets
import openai
import click
from tqdm import trange

openai.api_key = open("api_key.txt").read().strip()


def load_wikitext2() -> str:
    """Load the WikiText-2 dataset."""
    random.seed(42)
    data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    data = "\n".join(
        random.sample(
            [
                text.strip()
                for text in data["validation"]["text"]
                if len(text.strip()) != 0
            ],
            1000,
        )
    )
    return data


def lm_sentence_gpt3(sentence: str, engine: str) -> Tuple[List[str], List[float]]:
    """Evaluate the GPT-3 language model on a sentence."""
    response = openai.Completion.create(
        engine=engine,
        prompt=sentence,
        max_tokens=0,
        temperature=0,
        logprobs=1,
        stop=None,
        echo=True,
    )

    tokens = response["choices"][0]["logprobs"]["tokens"]
    logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
    return tokens, logprobs


@click.command()
@click.option("--engine", default="ada", help="GPT-3 engine to use")
def lm_gpt3(engine: str):
    data = load_wikitext2()
    data_tokens = data.split(" ")
    bptt = 512
    all_tokens, all_logprobs = [], []
    for i in trange(0, len(data_tokens), bptt):
        sentence = " ".join(data_tokens[i : i + bptt])
        tokens, logprobs = lm_sentence_gpt3(sentence, engine)
        all_tokens += tokens
        all_logprobs += logprobs
    json.dump([all_tokens, all_logprobs], open(f"gpt3_{engine}_wikitext2.json", "w"))


if __name__ == "__main__":
    lm_gpt3()
