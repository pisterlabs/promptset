import sys
import time
from utils import load_data, select_subset, get_instructions, act_icl
import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm
import sys
import argparse
import openai
from dotenv import load_dotenv
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=5)
parser.add_argument('--num_points', type=int, default=5)
parser.add_argument('--num_simulations', type=int, default=100)
parser.add_argument('--engine',  default='random')
args = parser.parse_args()  
num_features = args.num_features
num_points = args.num_points
num_simulations = args.num_simulations
engine = args.engine
llm = act_icl #TODO: Change this to your own LLM function
folder_name = f'data/{engine}_{num_features}features_{num_points}points'
#create folder if does not exist
os.makedirs(folder_name, exist_ok=True)

# Pre-processing of the tasks
lists_of_tasks =  ['pitcher', 'movie', 'whitewine', 'concrete', 'car', 'fish', 'occupation', 'medexp', 'tip', 'bone', 'recycle', 'plasma', 'prefecture', 'airfoil', 'mine', 'reactor', 'mammal', 'diabetes', 'air', 'vote', 'homeless', 'obesity', 'birthweight', 'algae', 'cigarette', 'schooling', 'mussel', 'bodyfat', 'sat', 'pinot', 'infant', 'lake', 'afl', 'mileage', 'wage', 'news', 'hitter', 'rent', 'men', 'rebellion', 'mortality', 'abalone', 'basketball', 'monet', 'athlete', 'excavator', 'contraception', 'home', 'laborsupply', 'dropout', 'cpu', 'fuel', 'land', 'highway', 'prostate', 'gambling', 'lung', 'crime', 'diamond', 'salary']
#Delete the tasks that have less than num_features features when loaded
idx = 0
while idx < len(lists_of_tasks):
    df = load_data(lists_of_tasks[idx], normalize_data=False)
    if df.shape[1] < num_features + 1:
        print("Deleting task " + lists_of_tasks[idx] + " because it has less than " + str(num_features) + " features")
        lists_of_tasks.pop(idx)
    else:
        idx += 1    
testing_tasks = lists_of_tasks.copy()

#Run the experiment
for task in tqdm(testing_tasks):
    #start equals the number of runs already done if the file exists
    start = pd.read_csv(folder_name + f'/{engine}_=' + task + '.csv')['run'].max() + 1 if os.path.exists(folder_name + f'/{engine}_=' + task + '.csv') else 0
    print(start)
    for i in range(start, num_simulations):
        prompt = get_instructions('generic')
        #testing
        data = []
        df = load_data(task, num_features)
        X, y, df_subset = select_subset(df, num_points)
        for n in range(num_points):
            x_vector = [df_subset[column].iloc[n] for column in df.columns[:-1]]
            prompt = prompt +  "x=[ " + ', '.join([str(int(x)) if x.is_integer() else str(x) for x in x_vector]) + "], y="
            ypred = act_icl(prompt, engine, sample_prior=n==0)
            row = [i, n, y[n], ypred, x_vector, task]
            data.append(row)
            prompt = prompt + " " + str(y[n])  + "\n"

        df_result = pd.DataFrame(data, columns=['run', 'trial', 'ytrue', 'ypred', 'x', 'name'])
        df_result.to_csv(folder_name + f'/{engine}_=' + task + '.csv', index=False, header=False if os.path.exists(folder_name + f'/{engine}_=' + task + '.csv') else True, mode='a')
