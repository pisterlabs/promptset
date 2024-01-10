import os
import re
import time
import random
import openai
from dotenv import load_dotenv
from datasets import load_dataset

from utils import *


def prompt_to_str(prev: str, prompt: dict) -> str:
    return prev + "Q: " + prompt["question"] + "\nA: " + prompt["answer"].replace("\n", " ") + "\n\n"


def ans_to_soln(answer: str) -> float:
    splits = answer.split("#### ")
    if len(splits) > 1:
        num = re.sub(r'[^0-9]', '', splits[1])
        if num:
            return float(num)
    return float("nan")


def prep_math_data(seed: int, num_prompts: int):
    random.seed(seed)
    dataset = load_dataset("gsm8k", "main")

    # simple:   4805 samples
    # medium:   3593 samples
    # hard:     394 samples
    # total:    8792 samples
    simple, medium, hard = [], [], []
    for split in dataset:
        for d in dataset[split]:
            steps = d["answer"].count("\n")
            if steps <= 3:
                d["diffifulty"] = "simple"
                simple.append(d)
            elif steps <= 6:
                d["diffifulty"] = "medium"
                medium.append(d)
            else:
                d["diffifulty"] = "hard"
                hard.append(d)
    total = simple[:300] + medium[:300] + hard[:300]

    simple_prompts = create_prompts(simple, num_prompts, prompt_to_str)
    medium_prompts = create_prompts(medium, num_prompts, prompt_to_str)
    hard_prompts = create_prompts(hard, num_prompts, prompt_to_str)
    print("----- SIMPLE -----\n", simple_prompts)
    print("----- MEDIUM -----\n", medium_prompts)
    print("----- HARD -----\n", hard_prompts)
    
    return total, simple_prompts, medium_prompts, hard_prompts


if __name__ == "__main__":
    SEED = 0
    NUM_PROMPTS = 6
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    RESULTS_PATH = os.path.join(ROOT_PATH, "results", "gsm8k")

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    (total,
     simple_prompts,
     medium_prompts,
     hard_prompts) = prep_math_data(SEED, NUM_PROMPTS)

    num_correct_simple = 0
    num_correct_medium = 0
    num_correct_hard = 0
    total_prompts = 0
    for i, p in enumerate(total):
        start = time.time()
        simple_correct = predict(
            simple_prompts, p, os.path.join(RESULTS_PATH, "results-simple.jsonl"), ans_to_soln)
        medium_correct = predict(
            medium_prompts, p, os.path.join(RESULTS_PATH, "results-medium.jsonl"), ans_to_soln)
        hard_correct = predict(
            hard_prompts, p, os.path.join(RESULTS_PATH, "results-hard.jsonl"), ans_to_soln)
        end = time.time()

        total_prompts += 1
        if simple_correct:
            num_correct_simple += 1
        if medium_correct:
            num_correct_medium += 1
        if hard_correct:
            num_correct_hard += 1

        print("Prompt #" + str(i) +
              f"\tSimple: {simple_correct}" +
              f"\tSimple Accuracy: {num_correct_simple}/{total_prompts} ({round(100 * num_correct_simple/total_prompts, 2)}%)" +
              f"\tMedium: {medium_correct}" +
              f"\tMedium Accuracy: {num_correct_medium}/{total_prompts} ({round(100 * num_correct_medium/total_prompts, 2)}%)" +
              f"\tHard: {hard_correct}" +
              f"\tHard Accuracy: {num_correct_hard}/{total_prompts} ({round(100 * num_correct_hard/total_prompts, 2)}%)" +
              f"\tTime: {round(end - start, 2)}")
