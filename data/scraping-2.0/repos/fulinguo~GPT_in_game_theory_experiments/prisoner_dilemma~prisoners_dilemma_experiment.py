import openai
import argparse
import json
import numpy as np
import re
import os
import time
from dotenv import load_dotenv, find_dotenv
import generate_prompt_template_pd
import random
import json

parser = argparse.ArgumentParser()

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


def get_response(messages, temperature, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format = {'type': 'json_object'},
    )
    return response['choices'][0]['message']['content']

def generate_system_message(feature):
    message = f"You are playing a multi-round game. You will be given instructions of the game. Important: Please pretend that you are a human in the game with the following features when making decisions: {feature}"
    return message


class PrisonersDilemma:
    def __init__(self, payoff_matrix, total_rounds, feature1, feature2, temperature, model):
        self.payoff_matrix = payoff_matrix
        self.total_rounds = total_rounds
        self.feature1 = feature1
        self.feature2 = feature2
        self.temperature = temperature
        self.model = model
        
    def run_prisoner_dilemma(self):

        round_history_1 = []
        round_history_2 = []
        results_strategies = np.zeros((self.total_rounds, 2))
        reasons = [[None]*2 for _ in range(self.total_rounds)]
        player1_earnings = 0
        player2_earnings = 0

        for round_num in range(1, self.total_rounds + 1):
            time.sleep(3)
            print(f"Round {round_num}")
            round_history_str1 = ' '.join(round_history_1)
            round_history_str2 = ' '.join(round_history_2)

            questionaire = random.choices([True, False], weights=[1., 0.])[0]

            player1_prompt = generate_prompt_template_pd.pd_prompt(total_rounds=self.total_rounds, round_num=round_num, feature=self.feature1,\
                                    round_history_str=round_history_str1, player_number=1, payoff_matrix=self.payoff_matrix, your_earnings=player1_earnings, \
                                    opponent_earnings=player2_earnings, questionaire=questionaire)
            
            player2_prompt = generate_prompt_template_pd.pd_prompt(total_rounds=self.total_rounds, round_num=round_num, feature=self.feature2,\
                                    round_history_str=round_history_str2, player_number=2, payoff_matrix=self.payoff_matrix, your_earnings=player2_earnings, \
                                    opponent_earnings=player1_earnings, questionaire=questionaire)
            
            system_message1 = generate_system_message(self.feature1)
            player1_messages = [{"role": "system", "content": system_message1}]
            player1_messages.append({"role": "user", "content": player1_prompt})
            
            system_message2 = generate_system_message(self.feature2)
            player2_messages = [{"role": "system", "content": system_message2}]
            player2_messages.append({"role": "user", "content": player2_prompt})

            if questionaire==False:
                response1 = get_response(player1_messages, self.temperature, self.model)
                print("response 1:", response1)
                strategy1_let = re.search(r'\b(cooperate|defect)\b', response1).group()
                
                response2 = get_response(player2_messages, self.temperature, self.model)
                print("response 2:", response2)
                strategy2_let = re.search(r'\b(cooperate|defect)\b', response2).group()
            else:
                response1 = get_response(player1_messages, self.temperature, self.model)
                print(player1_messages)
                print(response1)
                answer_dict1 = json.loads(response1)

                choice1 = answer_dict1['decision']
                reason1 = answer_dict1['reasoning']
                reasons[round_num-1][0] = reason1
                strategy1_let = re.search(r'\b(cooperate|defect)\b', choice1).group()
                print("choice1:", choice1)
                print(strategy1_let)

                response2 = get_response(player2_messages, self.temperature, self.model)
                print(player2_messages)
                print(response2)
                answer_dict2 = json.loads(response2)

                choice2 = answer_dict2['decision']
                reason2 = answer_dict2['reasoning']
                reasons[round_num-1][1] = reason2
                strategy2_let = re.search(r'\b(cooperate|defect)\b', choice2).group()
                print("choice2:", choice2)
                print(strategy2_let)
            
            
            if strategy1_let == 'cooperate':
                strategy1 = 0
            elif strategy1_let == 'defect':
                strategy1 = 1
            else:
                strategy1 = None

            if strategy2_let == 'cooperate':
                strategy2 = 0
            elif strategy2_let == 'defect':
                strategy2 = 1
            else:
                strategy2 = None


            try:
                payoff1 = self.payoff_matrix[strategy1][strategy2]
                payoff2 = self.payoff_matrix[strategy2][strategy1]
            except:
                return None

            player1_earnings += payoff1
            player2_earnings += payoff2

            round_summary_1 = f"Round {round_num} summary: [You choose to {strategy1_let} and the other player chooses to {strategy2_let}. You get {payoff1} dollars and the other player gets {payoff2} dollars ] \n"
            round_history_1.append(round_summary_1)

            round_summary_2 = f"Round {round_num} summary: [You choose to {strategy2_let} and the other player chooses to {strategy1_let}. You get {payoff2} dollars and the other player gets {payoff1} dollars ] \n"
            round_history_2.append(round_summary_2)

            results_strategies[round_num - 1][0] = strategy1
            results_strategies[round_num - 1][1] = strategy2

        return results_strategies, reasons
