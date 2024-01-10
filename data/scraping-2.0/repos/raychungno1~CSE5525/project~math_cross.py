import os
import time
import openai
from dotenv import load_dotenv

from utils import *
from maths import prep_math_data, ans_to_soln
from strat import prep_strat_data

if __name__ == "__main__":
    SEED = 0
    NUM_PROMPTS = 6
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    STRAT_DATA_PATH = os.path.join(ROOT_PATH, "data", "strategyqa_train.json")

    RESULTS_PATH = os.path.join(ROOT_PATH, "results", "gsm8k-cross")

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    (total, _, _, _) = prep_math_data(SEED, NUM_PROMPTS)
    (_,
     simple_prompts,
     medium_prompts,
     hard_prompts) = prep_strat_data(SEED, NUM_PROMPTS, STRAT_DATA_PATH)

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
