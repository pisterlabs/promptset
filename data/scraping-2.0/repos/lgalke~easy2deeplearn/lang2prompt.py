import os
import argparse
import glob
from typing import Generator

import pandas as pd

from language import load_input_language, Language
from learning_experiment import LearningExp


import openai

openai.organization = "org-MHqlKDvTANBEASvMZSUJGnvZ"
openai.api_key = os.getenv("OPENAI_API_KEY")


# Globals
PARTICIPANT_LOG_DIR = "data/"

OUTPUT_DIR = "gpt3-completions"

INSERTION_TOKEN = "[insert]"


def make_example(shape: int, angle: int, word: str = None) -> str:
    if word is None:
        # Placeholder for prediction
        word = INSERTION_TOKEN
    d = {"shape": shape, "angle": angle, "word": word}
    return str(d) + "\n"


def prepare_example_for_completion(example: str) -> str:
    """Use this function to prepare for the completion API instead of insertion"""
    return example[: example.find(INSERTION_TOKEN)]


def make_prompt(
    train_data: pd.DataFrame, mem_test: pd.DataFrame, gen_test: pd.DataFrame
) -> str:
    prompt = ""

    for __idx, row in train_data.iterrows():
        prompt += make_example(row.Shape, row.Angle, row.Word)

    prompt += "\n"

    for __idx, row in mem_test.iterrows():
        prompt += make_example(row.Shape, row.Angle)

    prompt += "\n"

    for __idx, row in gen_test.iterrows():
        prompt += make_example(row.Shape, row.Angle)

    return prompt


def query_gpt3_completion(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> pd.DataFrame:
    """Train data is a dataframe with Shape, Angle, and Word in each row
    Test data is a dataframe with Shape and Angle in each row, which can be either memorization or generalization data"""

    prompt_prefix = ""
    for __idx, row in train_data.iterrows():
        prompt_prefix += make_example(row.Shape, row.Angle, row.Word)

    predicted_words = []
    for __idx, row in test_data.iterrows():
        example = prepare_example_for_completion(make_example(row.Shape, row.Angle))
        prompt = prompt_prefix + example

        print("Prompt:", "[...]" + prompt.splitlines()[-1], sep="\n")
        result = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            stop="'",
        )
        print("Result:", result, sep="\n")
        predicted_word = result["choices"][0]["text"]
        predicted_words.append(predicted_word)

    results = test_data.copy(deep=True)
    # Drop unnecessary columns
    results.drop(
        ["SelectedItem"] + ["Distr%d" % i for i in range(1, 8)], axis=1, inplace=True
    )
    # Wipe previous results
    results.drop(["Input", "Correct", "Producer"], axis=1, inplace=True)
    results["Input"] = predicted_words
    results["Producer"] = "text-davinci-003"
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lang_id")
    parser.add_argument("--gpt3-mem-test", default=False, action="store_true")
    parser.add_argument("--gpt3-reg-test", default=False, action="store_true")
    args = parser.parse_args()

    # Find a matching logfile since we also need gen test data
    # But one is enough since all have the same
    pattern = os.path.join(
        PARTICIPANT_LOG_DIR, f"LearningExp_*_{args.lang_id}_*log.txt"
    )
    matching_logfiles = glob.glob(pattern)
    one_logfile = sorted(matching_logfiles)[0]

    lexp = LearningExp.load(one_logfile, with_input_language=True)
    lang = lexp.lang

    # train_data = lexp.get_all_training_data()
    # train_data["Word"] = train_data["Target"].map(lang.get_word_by_id)
    # print(train_data)
    train_data = lang.data

    mem_test_data = lexp.get_memorization_test_data()
    reg_test_data = lexp.get_regularization_test_data()

    if args.gpt3_mem_test or args.gpt3_reg_test:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.gpt3_mem_test:
        mem_results = query_gpt3_completion(train_data, mem_test_data)
        outfile = os.path.join(OUTPUT_DIR, f"gpt3-{lang.name}-mem-test.csv")
        mem_results.to_csv(outfile, index=False)

    if args.gpt3_reg_test:
        reg_results = query_gpt3_completion(train_data, reg_test_data)
        outfile = os.path.join(OUTPUT_DIR, f"gpt3-{lang.name}-reg-test.csv")
        reg_results.to_csv(outfile, index=False)

    if not args.gpt3_mem_test and not args.gpt3_reg_test:
        # Fallback print to console for copy-pasting the prompt manually
        prompt = make_prompt(
            train_data,
            mem_test_data,
            reg_test_data,
        )
        print(prompt)


if __name__ == "__main__":
    main()
