import openai
import argparse
import json
import numpy as np
import re
import os
import time
from dotenv import load_dotenv, find_dotenv
import generate_prompt_template_ug
import random
import json

parser = argparse.ArgumentParser()

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

def extract_json(answer):
    pattern = r'\{.*\}'
    
    match = re.search(pattern, answer)
    
    if match:
        json_string = match.group()
        json_string = re.sub(r'\s+', ' ', json_string)
        return json.loads(json_string)
    
    return {}



def get_response(messages, temperature, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        #response_format = {'type': 'json_object'},
    )
    return response['choices'][0]['message']['content']

def generate_system_message(feature):
    message = f"You are playing a multi-round game. You will be given instructions of the game. Important: Please pretend that you are a human in the game with the following features when making decisions: {feature}"
    return message

class UltimatumGame:
    def __init__(self, sum_of_money, total_rounds, feature1, feature2, temperature, model):
        self.sum_of_money = sum_of_money
        self.total_rounds = total_rounds
        self.feature1 = feature1
        self.feature2 = feature2
        self.temperature = temperature
        self.model = model


    def run_ultimatum_game(self):

        round_history_1 = []
        round_history_2 = []
        results_offer_act = np.zeros((self.total_rounds, 2))
        reasons = [[None]*2 for _ in range(self.total_rounds)]
        proposer_earnings = 0
        responder_earnings = 0

        for round_num in range(1, self.total_rounds+1):
            time.sleep(0)
            print(f"Round {round_num}")
            round_history_str1 = ' '.join(round_history_1)
            round_history_str2 = ' '.join(round_history_2)

            questionaire = random.choices([True, False], weights=[1., 0.])[0]

            proposer_prompt = generate_prompt_template_ug.ug_proposer_prompt(sum_of_money=self.sum_of_money, total_rounds=self.total_rounds,\
                    round_num=round_num, feature=self.feature1, round_history_str=round_history_str1,\
                          proposer_earnings=proposer_earnings, responder_earnings=responder_earnings, questionaire=questionaire)
            
            system_message1 = generate_system_message(self.feature1)
            proposer_messages = [{"role": "system", "content": system_message1}]
            proposer_messages.append({"role": "user", "content": proposer_prompt})
            print(proposer_messages)
            if questionaire==False:
                offer = get_response(proposer_messages, self.temperature, self.model)
                print("Offer:", offer)
            else:
                answer = get_response(proposer_messages, self.temperature, self.model)
                print(answer)
                answer_dict = extract_json(answer)

                offer = answer_dict['decision']
                reason = answer_dict['reasoning']
                reasons[round_num-1][0] = reason
                print("Offer:", offer)


            match = re.findall(r'\b\d+\b', offer)
            offered_amount = int(match[-1]) if match else None
            print(offered_amount)

            responder_prompt = generate_prompt_template_ug.ug_responder_prompt(sum_of_money=self.sum_of_money, total_rounds=self.total_rounds,\
                    round_num=round_num, feature=self.feature2, round_history_str=round_history_str2, proposer_earnings=proposer_earnings,\
                          responder_earnings=responder_earnings, offered_amount=offered_amount, questionaire=questionaire)
            
            system_message2 = generate_system_message(self.feature2)
            responder_messages = [{"role": "system", "content": system_message2}]
            responder_messages.append({"role": "user", "content": responder_prompt})
            print(responder_messages)

            if questionaire==False:
                decision = get_response(responder_messages, self.temperature, self.model)
                print("Decision:", decision)
            else:
                answer = get_response(responder_messages, self.temperature, self.model)
                print(answer)
                answer_dict = extract_json(answer)
                decision = answer_dict['decision']
                reason = answer_dict['reasoning']
                reasons[round_num-1][1] = reason
                print("Decision:", decision)


            round_summary_1 = f"Round {round_num} summary: [You keep {self.sum_of_money - offered_amount} dollars yourself and offer {offered_amount} dollars to the responder; Decision by the responder: {decision}] \n"
            round_history_1.append(round_summary_1)
            round_summary_2 = f"Round {round_num} summary: [The proposer keeps {self.sum_of_money - offered_amount} dollars himself and offers {offered_amount} dollars to you; Decision by you: {decision}] \n"
            round_history_2.append(round_summary_2)

            results_offer_act[round_num-1][0] = offered_amount
            results_offer_act[round_num-1][1] = 1 if 'accept' in decision.lower() else 0
            print(results_offer_act[round_num-1][1])
            print("\n")

            if 'accept' in decision.lower():
                proposer_earnings += self.sum_of_money - offered_amount
                responder_earnings += offered_amount


            print(proposer_earnings)
            print(responder_earnings)
                

        return results_offer_act, reasons


