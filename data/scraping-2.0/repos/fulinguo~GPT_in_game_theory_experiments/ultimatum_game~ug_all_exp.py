import openai
import argparse
import json
import numpy as np
import re
import os
import time
from dotenv import load_dotenv, find_dotenv
import generate_prompt_template_ug
from ultimatum_game_experiment import UltimatumGame
import random
import json

parser = argparse.ArgumentParser()

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

parser.add_argument("--sum_of_money", type=float, default=100)
parser.add_argument("--total_rounds", type=int, default=5)
parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()
sum_of_money = args.sum_of_money
total_rounds = args.total_rounds
model = args.model
temperature = args.temperature

current_directory = os.getcwd()

four_types = ['SF', 'FS', 'SS', 'FF']

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
        all_results = np.load(os.path.join(current_directory, 'ultimatum_game', 'exp_data', f"results_ug_{type_comb}.npy"))
        all_reasons = np.load(os.path.join(current_directory, 'ultimatum_game', 'exp_data', f"reasons_ug_{type_comb}.npy"), allow_pickle=True)
        
        print(all_results.shape)
        print(all_reasons.shape)
    except:
        all_results = np.empty((0, total_rounds, 2))
        all_reasons = np.empty((0, total_rounds, 2), dtype=object)

    # Run simulations and store results
    for i in range(num_simulations):
        try:
            game = UltimatumGame(sum_of_money=sum_of_money, total_rounds=total_rounds, feature1=feature1, feature2=feature2, temperature=temperature, model=model)
            result, reasons = game.run_ultimatum_game()
            all_results = np.append(all_results, [result], axis=0)
            all_reasons = np.append(all_reasons, [reasons], axis=0)
            np.save(os.path.join(current_directory, 'ultimatum_game', 'exp_data', f"results_ug_{type_comb}.npy"), all_results)
            np.save(os.path.join(current_directory, 'ultimatum_game', 'exp_data', f"reasons_ug_{type_comb}.npy"), all_reasons)
        except:
            print(f"Error in simulation {i}")
            continue

    print(all_results, reasons)


    
