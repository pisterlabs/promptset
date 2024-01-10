import openai
import argparse
import json
import numpy as np
import re
import os
import time
from dotenv import load_dotenv, find_dotenv
import generate_prompt_template_pd
from prisoners_dilemma_experiment import PrisonersDilemma
import random


parser = argparse.ArgumentParser()

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

parser.add_argument("--total_rounds", type=int, default=5)
parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()
total_rounds = args.total_rounds
model = "gpt-4-1106-preview"
temperature = 1.0
payoff_matrix = np.array([[200, 0], \
                          [300, 100]])


current_directory = os.getcwd()
num_simulations = 1
type_comb = 'FS'

if __name__ == "__main__":
    if type_comb == 'SF':
        feature1 = 'payoff maximization, strategic thinking, selfishness'
        feature2 = 'payoff maximization, strategic thinking, fairness concern'

    elif type_comb == 'FS':
        feature1 = 'payoff maximization, strategic thinking, fairness concern'
        feature2 = 'payoff maximization, strategic thinking, selfishness'

    elif type_comb == 'SS':
        feature1 = 'payoff maximization, strategic thinking, selfishness'
        feature2 = 'payoff maximization, strategic thinking, selfishness'

    elif type_comb == 'FF':
        feature1 = 'payoff maximization, strategic thinking, fairness concern'
        feature2 = 'payoff maximization, strategic thinking, fairness concern'

    try:
        all_results = np.load(os.path.join(current_directory, 'prisoner_dilemma', 'exp_data', f"results_pd_{type_comb}.npy"))
        all_reasons = np.load(os.path.join(current_directory, 'prisoner_dilemma', 'exp_data', f"reasons_pd_{type_comb}.npy"), allow_pickle=True)
        print(all_results.shape)
        print(all_reasons.shape)
    except FileNotFoundError:
        all_results = np.empty((0, total_rounds, 2))
        all_reasons = np.empty((0, total_rounds, 2), dtype=object)

    # Run simulations and store results
    for i in range(num_simulations):
        time.sleep(0)
        try:
            game = PrisonersDilemma(payoff_matrix=payoff_matrix, total_rounds=total_rounds, feature1=feature1, feature2=feature2, temperature=temperature, model=model)
            result, reasons = game.run_prisoner_dilemma()
            all_results = np.append(all_results, [result], axis=0)
            all_reasons = np.append(all_reasons, [reasons], axis=0)

            np.save(os.path.join(current_directory, 'prisoner_dilemma', 'exp_data', f"results_pd_{type_comb}.npy"), all_results)
            np.save(os.path.join(current_directory, 'prisoner_dilemma', 'exp_data', f"reasons_pd_{type_comb}.npy"), all_reasons)
        except Exception as e:
            print(e)
            print("Error in simulation")
            continue

    print(all_results, reasons)

    


