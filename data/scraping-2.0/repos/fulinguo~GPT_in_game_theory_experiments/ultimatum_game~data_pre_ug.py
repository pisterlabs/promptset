import json
import numpy as np
import re
import os
import time
import openai

import argparse
from dotenv import load_dotenv, find_dotenv

parser = argparse.ArgumentParser()

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


current_directory = os.getcwd()
current_directory = os.path.join(current_directory, 'ultimatum_game')


results_array_FF = np.load(os.path.join(current_directory, 'exp_data', "results_ug_FF.npy"))
reasons_array_FF = np.load(os.path.join(current_directory, 'exp_data', "reasons_ug_FF.npy"), allow_pickle=True)

results_array_SS = np.load(os.path.join(current_directory, 'exp_data', "results_ug_SS.npy"))
reasons_array_SS = np.load(os.path.join(current_directory, 'exp_data', "reasons_ug_SS.npy"), allow_pickle=True)

results_array_FS = np.load(os.path.join(current_directory, 'exp_data', "results_ug_FS.npy"))
reasons_array_FS = np.load(os.path.join(current_directory, 'exp_data', "reasons_ug_FS.npy"), allow_pickle=True)

results_array_SF = np.load(os.path.join(current_directory, 'exp_data', "results_ug_SF.npy"))
reasons_array_SF = np.load(os.path.join(current_directory, 'exp_data', "reasons_ug_SF.npy"), allow_pickle=True)

all_treat = ['FF', 'SS', 'FS', 'SF']
results_array = {'FF': results_array_FF, 'SS': results_array_SS, 'FS': results_array_FS, 'SF': results_array_SF}
reasons_array = {'FF': reasons_array_FF, 'SS': reasons_array_SS, 'FS': reasons_array_FS, 'SF': reasons_array_SF}

sum_of_money = 100
rounds_num = 5

colors = {'FF': 'blue', 'SS': 'red', 'FS': 'green', 'SF': 'black'}
names = {'FF': 'fair-fair (FF)', 'SS': 'selfish-selfish (SS)', 'FS': 'fair-selfish (FS)', 'SF': 'selfish-fair (SF)'}

print(len(results_array_FF))
print(len(results_array_SS))
print(len(results_array_FS))
print(len(results_array_SF))


# for each treatment, generate a dataframe with the following columns: round, offer, acceptance, reason for offer, reason for acceptance
import pandas as pd

def generate_df(treat):
    num_simulations, num_rounds, _ = results_array[treat].shape
    round_array = []
    offer_array = []
    acceptance_array = []
    reason_offer_array = []
    reason_acceptance_array = []
    
    for sim in range(num_simulations):
        for t in range(num_rounds):
            round_array.append(t+1)
            offer_array.append(results_array[treat][sim, t, 0])
            acceptance_array.append(results_array[treat][sim, t, 1])
            reason_offer_array.append(reasons_array[treat][sim, t, 0])
            reason_acceptance_array.append(reasons_array[treat][sim, t, 1])

    acceptance_array = np.array(acceptance_array)
    
    df = pd.DataFrame({'round': round_array, 'offer': offer_array, 'acceptance': acceptance_array, 'rejection': 1-acceptance_array, 'reason_proposer': reason_offer_array, 'reason_responder': reason_acceptance_array})
    return df


# if df_FF not exists, generate it
if not os.path.exists(os.path.join(current_directory, 'exp_data', 'df_FF.csv')):
    df_FF = generate_df('FF')
    df_FF.to_csv(os.path.join(current_directory, 'exp_data', 'df_FF.csv'))
    df_SS = generate_df('SS')
    df_SS.to_csv(os.path.join(current_directory, 'exp_data', 'df_SS.csv'))
    df_FS = generate_df('FS')
    df_FS.to_csv(os.path.join(current_directory, 'exp_data', 'df_FS.csv'))
    df_SF = generate_df('SF')
    df_SF.to_csv(os.path.join(current_directory, 'exp_data', 'df_SF.csv'))
    df_FF['treatment'] = 'FF'
    df_SS['treatment'] = 'SS'
    df_FS['treatment'] = 'FS'
    df_SF['treatment'] = 'SF'

    df_all = pd.concat([df_FF, df_SS, df_FS, df_SF])

    # create a new column to indicate whether the proposer and the responder are selfish or fair
    df_all['proposer_selfish'] = df_all['treatment'].apply(lambda x: 'selfish' if x == 'SS' or x == 'SF' else 'fair')
    df_all['responder_selfish'] = df_all['treatment'].apply(lambda x: 'selfish' if x == 'FS' or x == 'SS' else 'fair')

    # save df_all as a csv file
    df_all.to_csv(os.path.join(current_directory, 'exp_data', 'df_all.csv'))

else:
    print('Dataframes already exist')
    df_all = pd.read_csv(os.path.join(current_directory, 'exp_data', "df_all.csv"), index_col=0)
    df_FF = pd.read_csv(os.path.join(current_directory, 'exp_data', "df_FF.csv"), index_col=0)
    df_SS = pd.read_csv(os.path.join(current_directory, 'exp_data', "df_SS.csv"), index_col=0)
    df_FS = pd.read_csv(os.path.join(current_directory, 'exp_data', "df_FS.csv"), index_col=0)
    df_SF = pd.read_csv(os.path.join(current_directory, 'exp_data', "df_SF.csv"), index_col=0)


import openai

df_all = df_all.reset_index(drop=True)

def prompt_template_for_reasoning(player, category_list, additional_text):
    example_output = '{'
    for category in category_list:
        example_output += f'"{category}": 1, '
    example_output = example_output[:-2]
    example_output += '}'

    text = f"""
You are a helpful assistant answering whether a given reasoning statement provided by a {player} in a multi-round Ultimatum Game indicates each of the following categories is contained in the reasoning: {category_list}.

{additional_text}

Please output a single-line JSON object without line-break or any other words, where the key is each category name and the value is either 1 or 0, where 1 means the reasoning statement contains the category and 0 means otherwise.

The reasoning statement can belong to multiple categories.

The reasoning statement will be given within a square bracket.

Example output: {example_output}

Please answer in the exact format.

"""
    return text
    

temperature = 0.
model = "gpt-4-1106-preview"

def get_response(messages, temperature, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format = {'type': 'json_object'},
    )
    return response['choices'][0]['message']['content']


category_list_accept_responder = ['accept_get_payoff', 'expect_better_future', 'limited_rounds']
if 'responder_text_accept_analysed' not in df_all.columns:
    df_all['responder_text_accept_analysed'] = False

    for feature in category_list_accept_responder:
        df_all[feature+'_responder'] = np.nan

additional_text_responder_accept = '''
accept_get_payoff: The responder mentions that accepting the offer will get them a payoff, or mentions that rejecting will result in no payoff.
expect_better_future: The responder mentions that they expect better offers in the future or they may make different decisions in the future.
limited_rounds: The responder mentions that there are limited rounds in the game, so they may accept the offer to get a payoff.
'''

RUN_GPT = False


if RUN_GPT:
    # Get the reasoning statement for the responder and update the feature columns
    for index, row in df_all.iterrows():
        print(index)
        column_list = [row[feature+'_responder'] for feature in category_list_accept_responder]

        if np.isnan(column_list).any():
            if row['acceptance'] == 0:
                continue
            if row['offer'] > 30:
                continue

            # sleep for 1 second
            time.sleep(1)
            category_list = category_list_accept_responder

            system_prompt = prompt_template_for_reasoning('responder', category_list, additional_text_responder_accept)
            reason_statement = row['reason_responder']
            user_prompt = f"The reasoning statement is shown within the square bracket []: [{reason_statement}]"

            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            try:
                response = get_response(message, temperature, model)
                response_dict = json.loads(response)
                for feature in category_list:
                    df_all.at[index, feature+'_responder'] = response_dict[feature]

                df_all.at[index, 'responder_text_accept_analysed'] = True
            except Exception as e:
                print(f"Error in processing row {index}: {e}")

        else:
            print(f'row {index} has been analysed')

        df_all.to_csv(os.path.join(current_directory, 'exp_data', 'df_all.csv'))


category_list_reject_responder_3 = ["consistently_diminishing_offers" , "potential_for_higher_future_offer"]



if 'responder_text_reject_analysed_3' not in df_all.columns:
    df_all['responder_text_reject_analysed_3'] = False

    for feature in category_list_reject_responder_3:
        df_all[feature+'_responder'] = np.nan

additional_text_responder_reject_3 = '''
consistently_diminishing_offers: The responder mentions that the proposer's offer is consistently diminishing or less fair.
potential_for_higher_future_offer: The responder mentions that they expect higher or fairer offers in the future.
'''

RUN_GPT = True

if RUN_GPT:
    # Get the reasoning statement for the responder and update the feature columns
    for index, row in df_all.iterrows():
        print(index)
        column_list = [row[feature+'_responder'] for feature in category_list_reject_responder_3]

        if np.isnan(column_list).any():
            if row['acceptance'] == 1:
                continue
            if row['round'] != 3:
                continue

            # sleep for 1 second
            time.sleep(1)
            category_list = category_list_reject_responder_3

            system_prompt = prompt_template_for_reasoning('responder', category_list, additional_text_responder_reject_3)
            reason_statement = row['reason_responder']
            user_prompt = f"The reasoning statement is shown within the square bracket []: [{reason_statement}]"

            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            try:
                response = get_response(message, temperature, model)
                response_dict = json.loads(response)
                for feature in category_list:
                    df_all.at[index, feature+'_responder'] = response_dict[feature]

                df_all.at[index, 'responder_text_reject_analysed_3'] = True
            except Exception as e:
                print(f"Error in processing row {index}: {e}")

        else:
            print(f'row {index} has been analysed')

        df_all.to_csv(os.path.join(current_directory, 'exp_data', 'df_all.csv'))



print(df_all)