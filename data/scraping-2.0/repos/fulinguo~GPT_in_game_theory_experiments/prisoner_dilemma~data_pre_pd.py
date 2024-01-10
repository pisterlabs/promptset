import json
import numpy as np
import re
import os
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from dotenv import load_dotenv, find_dotenv
import openai

parser = argparse.ArgumentParser()

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

current_directory = os.getcwd()
current_directory = os.path.join(current_directory, 'prisoner_dilemma')

results_array_FF = np.load(os.path.join(current_directory, 'exp_data', "results_pd_FF.npy"))
reasons_array_FF = np.load(os.path.join(current_directory, 'exp_data', "reasons_pd_FF.npy"), allow_pickle=True)

results_array_FS = np.load(os.path.join(current_directory, 'exp_data', "results_pd_FS.npy"))
reasons_array_FS = np.load(os.path.join(current_directory, 'exp_data', "reasons_pd_FS.npy"), allow_pickle=True)

results_array_SS = np.load(os.path.join(current_directory, 'exp_data', "results_pd_SS.npy"))
reasons_array_SS = np.load(os.path.join(current_directory, 'exp_data', "reasons_pd_SS.npy"), allow_pickle=True)


rounds_num = 5

all_treat = ['FF', 'SS', 'FS']
colors = {'FF': 'blue', 'SS': 'red', 'FS': 'g'}
names = {'FF': 'fair-fair (FF)', 'SS': 'selfish-selfish (SS)', 'FS': 'fair-selfish (FS)'}
results_array = {'FF': results_array_FF, 'SS': results_array_SS, 'FS': results_array_FS}
reasons_array = {'FF': reasons_array_FF, 'SS': reasons_array_SS, 'FS': reasons_array_FS}


all_treat2 = ['FwF', 'FwS', 'SwS', 'SwF']
colors2 = {'FwF': 'blue', 'SwS': 'red', 'FwS': 'green', 'SwF': 'black'}
names2 = {'FwF': 'fair (with fair)', 'SwS': 'selfish (with selfish)', 'FwS': 'fair (with selfish)', 'SwF': 'selfish (with fair)'}


print(len(results_array_FF))
print(len(results_array_SS))
print(len(results_array_FS))

print(len(reasons_array_FF))
print(len(reasons_array_SS))
print(len(reasons_array_FS))

# create a dataframe extracting the data for fair and selfish players, with column simulation_num, own_feature, other_feature, treatment, 
# round, choice, other_choice reason, last_own_choice, last_other_choice, last_AA, last_BB, last_AB, last_BA

# check if df_all.csv exists in the directory

if os.path.exists(os.path.join(current_directory, 'exp_data', "df_all.csv")):
    df_all = pd.read_csv(os.path.join(current_directory, 'exp_data', "df_all.csv"))
    print('df_all.csv exists in the directory')
    print('df_all.csv has {} rows'.format(len(df_all)))

else:
    print('df_all.csv does not exist in the directory')
    id_array = []
    simulation_num_array = []
    rounds_array = []
    choice_array = []
    reason_array = []
    treatment_array = []
    own_feature_array = []
    other_feature_array = []
    other_choice_array = []
    last_own_choice_array = []
    last_other_choice_array = []


    for treatment in all_treat:
        for i in range(len(results_array[treatment])):
            for j in range(rounds_num):
                id_array.append(i*1000 + j*100 + 1)
                simulation_num_array.append(i)
                rounds_array.append(j+1)
                treatment_array.append(treatment)

                if j == 0:
                    last_own_choice_array.append(-1)
                    last_other_choice_array.append(-1)
                else:
                    last_own_choice_array.append(results_array[treatment][i][j-1][0])
                    last_other_choice_array.append(results_array[treatment][i][j-1][1])
                
                if treatment == 'FF':
                    own_feature_array.append('fair')
                    other_feature_array.append('fair')
                    choice_array.append(results_array[treatment][i][j][0])
                    reason_array.append(reasons_array[treatment][i][j][0])
                    other_choice_array.append(results_array[treatment][i][j][1])

                
                elif treatment == 'SS':
                    own_feature_array.append('selfish')
                    other_feature_array.append('selfish')
                    choice_array.append(results_array[treatment][i][j][0])
                    reason_array.append(reasons_array[treatment][i][j][0])
                    other_choice_array.append(results_array[treatment][i][j][1])

                elif treatment == 'FS':
                    own_feature_array.append('fair')
                    other_feature_array.append('selfish')
                    choice_array.append(results_array[treatment][i][j][0])
                    reason_array.append(reasons_array[treatment][i][j][0])
                    other_choice_array.append(results_array[treatment][i][j][1])

    for treatment in all_treat:
        for i in range(len(results_array[treatment])):
            for j in range(rounds_num):
                id_array.append(i*1000 + j*100 + 1)
                simulation_num_array.append(i)
                rounds_array.append(j+1)
                treatment_array.append(treatment)

                if j == 0:
                    last_own_choice_array.append(-1)
                    last_other_choice_array.append(-1)
                else:
                    last_own_choice_array.append(results_array[treatment][i][j-1][1])
                    last_other_choice_array.append(results_array[treatment][i][j-1][0])
                
                if treatment == 'FF':
                    own_feature_array.append('fair')
                    other_feature_array.append('fair')
                    choice_array.append(results_array[treatment][i][j][1])
                    reason_array.append(reasons_array[treatment][i][j][1])
                    other_choice_array.append(results_array[treatment][i][j][0])

                
                elif treatment == 'SS':
                    own_feature_array.append('selfish')
                    other_feature_array.append('selfish')
                    choice_array.append(results_array[treatment][i][j][1])
                    reason_array.append(reasons_array[treatment][i][j][1])
                    other_choice_array.append(results_array[treatment][i][j][0])

                elif treatment == 'FS':
                    own_feature_array.append('selfish')
                    other_feature_array.append('fair')
                    choice_array.append(results_array[treatment][i][j][1])
                    reason_array.append(reasons_array[treatment][i][j][1])
                    other_choice_array.append(results_array[treatment][i][j][0])

    # create the dataframe
    import pandas as pd
    df_all = pd.DataFrame()
    df_all['simulation_num'] = simulation_num_array
    df_all['round'] = rounds_array
    df_all['choice'] = choice_array
    df_all['reason'] = reason_array
    df_all['treatment'] = treatment_array
    df_all['own_feature'] = own_feature_array
    df_all['other_feature'] = other_feature_array
    df_all['other_choice'] = other_choice_array
    df_all['last_own_choice'] = last_own_choice_array
    df_all['last_other_choice'] = last_other_choice_array


    # dummy column last_AA, last_BB, last_AB, last_BA
    df_all['last_AA'] = 0
    df_all['last_BB'] = 0
    df_all['last_AB'] = 0
    df_all['last_BA'] = 0

    df_all.loc[(df_all['last_own_choice'] == 0) & (df_all['last_other_choice'] == 0), 'last_AA'] = 1
    df_all.loc[(df_all['last_own_choice'] == 1) & (df_all['last_other_choice'] == 1), 'last_BB'] = 1
    df_all.loc[(df_all['last_own_choice'] == 0) & (df_all['last_other_choice'] == 1), 'last_AB'] = 1
    df_all.loc[(df_all['last_own_choice'] == 1) & (df_all['last_other_choice'] == 0), 'last_BA'] = 1

    df_all.to_csv(os.path.join(current_directory, 'exp_data', "df_all.csv"), index=False)



print(df_all.columns)


df_all = df_all.reset_index(drop=True)

def prompt_template_for_reasoning(category_list, additional_text):
    example_output = '{'
    for category in category_list:
        example_output += f'"{category}": 1, '
    example_output = example_output[:-2]
    example_output += '}'

    text = f"""
You are a helpful assistant answering whether a given reasoning statement in a multi-round Prisoner's Dilemma contains information of each of the following categories: {category_list}.

{additional_text}

Please output a single-line JSON object without line-break or any other words, where the key is each category name and the value is either 1 or 0, where 1 means the reasoning statement contains information of the category and 0 means the reasoning statement does not contain information of the category.

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



category_list_pd_3 = ['reputation_building', 'altruism']

additional_text_pd_3 = '''
'reputation_building': Motivated by the desire to build a reputation as a cooperative player, which means that the player thinks that his own cooperation in this round may lead to the other player's cooperation in the future and thus lead to mutual cooperation and higher payoff.
'altruism': Motivated by the desire to help the other player, which means that the player cares about the other player's payoff and does not want to take advantage of the other player.
'''

if 'pd_text_coop_repu_alt_analysed' not in df_all.columns:
    df_all['pd_text_coop_repu_alt_analysed'] = False
    for feature in category_list_pd_3:
        df_all[feature+'_pd'] = np.nan

RUN_GPT = False

import time

if RUN_GPT:
            
    for index, row in df_all.iterrows():                
        print(index)
                    
        if row['choice'] == 1:
            continue

        if row['round']==5:
            continue
        
        if row['pd_text_coop_repu_alt_analysed']:
            continue
            
        time.sleep(2)
                    
        category_list = category_list_pd_3
        system_prompt = prompt_template_for_reasoning(category_list, additional_text_pd_3)
        reason_statement = row['reason']
        user_prompt = f"The reasoning statement is shown within the square bracket []: [{reason_statement}]"
            
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            response = get_response(message, temperature, model)
            response_dict = json.loads(response)
            for feature in category_list:
                df_all.at[index, feature+'_pd'] = response_dict[feature]
    
            print(response)
            print(reason_statement)
        
            df_all.at[index, 'pd_text_coop_repu_alt_analysed'] = True
        except Exception as e:
            print(f"Error in processing row {index}: {e}")
            
        df_all.to_csv(os.path.join(current_directory, 'exp_data', "df_all.csv"), index=False)



def prompt_template_for_one_class(additional_text):
    
    text = f"""
You are a helpful assistant answering whether a given reasoning statement provided by a player in a Prisoner's Dilemma indicates that they made a mistake in reasoning.

{additional_text}

Please output one of the following options: yes, no, unsure.

The reasoning statement will be given within a square bracket.

Please just answer one word in the exact format without any other words.

"""
    return text


if 'pd_text_error_analysed' not in df_all.columns:
    df_all['pd_text_error_analysed'] = False
    df_all['error_pd'] = np.nan

additional_text_pd_error = '''
The mistakes include the following:
- The player thought that his own defection in this round may lead to another player defecting (which is not true because they make decisions simultaneously).
- The player thought that mutual defection is a risk (which is not true because defection is a dominant strategy).
- The player thought his behaviour in this round may lead to retaliation in the future (which is not true because the game ends after this round).

If any of the above mistakes is made, please answer yes.
If none of the above mistakes is made, please answer no.
If you think it is not clear whether the player made a mistake, please answer unsure.
'''

def get_response(messages, temperature, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        #response_format = {'type': 'json_object'},
    )
    return response['choices'][0]['message']['content']

RUN_GPT = True

import time

if RUN_GPT:
    
        for index, row in df_all.iterrows():
            
            print(index)
    
            if row['choice'] == 1:
                continue

            if row['round']<5:
                continue
    
            if row['pd_text_error_analysed']:
                continue
    
            # sleep for 1 second
            time.sleep(1)
            
            system_prompt = prompt_template_for_one_class(additional_text_pd_error)
            reason_statement = row['reason']
            user_prompt = f"The reasoning statement is shown within the square bracket []: [{reason_statement}]"
    
            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            try:
                response = get_response(message, temperature, model)
                if response == 'yes':
                    response_num = 1
                elif response == 'no':
                    response_num = 0
                elif response == 'unsure':
                    response_num = np.nan
                df_all.at[index, 'error_pd'] = response_num
                df_all.at[index, 'pd_text_error_analysed'] = True

                print(response)
                print(reason_statement)
            except Exception as e:
                print(f"Error in processing row {index}: {e}")
    
            df_all.to_csv(os.path.join(current_directory, 'exp_data', "df_all.csv"), index=False)

print(df_all)






