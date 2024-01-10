import os
import time
import openai
import json

SEED = 0
NUM_PROMPTS = 6
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MATH_RESULTS_PATH = os.path.join(ROOT_PATH, "results", "gsm8k-cross")
STRAT_RESULTS_PATH = os.path.join(ROOT_PATH, "results", "strategyqa-cross")

for diffifulty in ["simple", "medium", "hard"]:
    FILE_PATH = os.path.join(MATH_RESULTS_PATH, f"results-{diffifulty}.jsonl")
    MATH_FILE_PATH = os.path.join(
        MATH_RESULTS_PATH, f"new-results-{diffifulty}.jsonl")
    STRAT_FILE_PATH = os.path.join(
        STRAT_RESULTS_PATH, f"new-results-{diffifulty}.jsonl")

    old_file = list(open(FILE_PATH, "r"))
    math_file = open(MATH_FILE_PATH, "w")
    strat_file = open(STRAT_FILE_PATH, "w")

    for i, line in enumerate(old_file):
        if i <= 8100:
            math_file.write(line)
        else:
            strat_file.write(line)
