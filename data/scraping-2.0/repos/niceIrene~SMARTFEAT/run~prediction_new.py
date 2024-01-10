import sys
print(sys.executable)
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
sys.path.append("./")
sys.path.append('../prompts/')
import pathlib as Path
from SMARTFEAT.serialize import *
import os
import argparse
import openai
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory
from SMARTFEAT.search import CurrentAttrLst
import pandas as pd 
import numpy as np
from SMARTFEAT.Prediction_helper import *
from sklearn.model_selection import train_test_split
from SMARTFEAT.search import *
import copy
from SMARTFEAT.feature_evaluation import *
import time


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type= str, default='./dataset/pima_diabetes/')
    args.add_argument('--predict_col', type=str, default='Outcome')
    args.add_argument('--csv', type=str, default='diabetes.csv')
    # args.add_argument('--model', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003'], default='gpt-3.5-turbo')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--n_generate_sample', type=int, default=1)
    args.add_argument('--clf_model', type=str, default='Decision Tree')
    args.add_argument('--delimiter', type=int, default=1)
    args.add_argument('--sampling_budget', type=int, default=10)
    args = args.parse_args()
    return args

args = parse_args()

if args.delimiter ==1:
    data_df = pd.read_csv(args.path + args.csv, delimiter=',')
else:
    data_df = pd.read_csv(args.path + args.csv, delimiter=';')

attributes = list(data_df.columns)
attributes.remove(args.predict_col)
org_features = attributes
with open(args.path+ 'data_agenda.txt', 'r') as f:
    data_agenda = f.read()
f.close

# drop the index column if have
for c in list(org_features):
    if 'Unnamed' in c:
        org_features.remove(c)

print("The original features are")
print(org_features)

# initialize the root state 
cur_attr_lst = CurrentAttrLst(org_features, data_agenda, data_df, args.clf_model, args.sampling_budget)

while True:
    try:
        if cur_attr_lst.step < len(org_features):
            result_lst = -1
            result_lst = feature_generator_propose(cur_attr_lst, org_features, args.predict_col)
            cur_attr_lst.step += 1
            if result_lst is None:
                continue
            for r in result_lst:
                print("Start value evaluation")
                print(r)
                state_evaluator(r, cur_attr_lst, args.predict_col)
        else:
            result_lst = feature_genetor_sampling(cur_attr_lst, args.predict_col, args.temperature, args.n_generate_sample)
            if result_lst == -1 or cur_attr_lst.budget_cur > cur_attr_lst.budget:
                cur_attr_lst.budget_cur = 0
                # more than five continuous failures or reach budget
                if isinstance(cur_attr_lst.last_op, MultiExtractor):
                    print("Search process ends")
                    break
                elif isinstance(cur_attr_lst.last_op, BinaryOperatorAlter):
                    # for binary operator reaches the generation error times.
                    print("Binary ends, go to aggregator")
                    cur_attr_lst.previous = 0
                    cur_attr_lst.last_op = AggregateOperator(cur_attr_lst.data_agenda, cur_attr_lst.model, args.predict_col, args.n_generate_sample)
                    continue
                elif isinstance(cur_attr_lst.last_op, AggregateOperator):
                    # for binary operator reaches the generation error times.
                    print("Aggregate ends, go to extract")
                    cur_attr_lst.previous = 0
                    cur_attr_lst.last_op = MultiExtractor(cur_attr_lst.data_agenda, cur_attr_lst.model, args.predict_col)
                    continue
            elif result_lst is None or len(result_lst) == 0:
                print("result lst is empty")
                cur_attr_lst.previous += 1
                continue
            else:
                for r in result_lst:
                    if cur_attr_lst.budget_cur <= cur_attr_lst.budget and cur_attr_lst.previous < 5:
                        state_evaluator(r, cur_attr_lst, args.predict_col)
                    else:
                        print("Budget or error times reached!!!!")
                        break
    except Exception as e:
        print("Exception occurs!!!")
        print(e)
        wait_time = 2  # Delay in seconds
        time.sleep(wait_time)
        continue
# lastly drop columns
cols_to_drop = list(set(cur_attr_lst.unary_attr) - set(cur_attr_lst.other_attr))
print("Columns to drop is")
print(cols_to_drop)
cur_attr_lst.df = cur_attr_lst.df.drop(columns = cols_to_drop, axis=1)
print(cur_attr_lst.df.head())
cur_attr_lst.df.to_csv("current_df_final.csv")