
import pandas as pd

import os
import openai
from datetime import datetime

import json
import time


log_dir = '' # log directory
# get current date string in the format of YYYYMMDD
date_string = datetime.now().strftime("%Y%m%d")
log_file = log_dir + 'log_'+date_string+'.csv'


data_dir = '' # data directory

input_file = data_dir + '2.all_questions_ similarity_scores.csv'
output_file = data_dir + '3.labeled_data_similarity_scores.csv'


# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# read the input score file
df_output = pd.read_csv(output_file, encoding='utf-8')
df_input = pd.read_csv(input_file, encoding='utf-8')

# lower case the question keyword column of the dataframes
df_output['question_keyword'] = df_output['question_keyword'].str.lower()
df_input['question_keyword'] = df_input['question_keyword'].str.lower()


# loop through the dataframe
for index, row in df_output.iterrows():
    id = row['id']
    question_keyword = row['question_keyword']

    question_approach = 'Challenge with how'

    # get the consistency score from the consistency dataframe with id, keyword and approach
    df_consistency_filtered = df_input[(df_input['id'] == id) & (df_input['question_keyword'] == question_keyword) & (df_input['question_approach'] == question_approach)]

    assert len(df_consistency_filtered) == 1


    techniques = ["", "_qaqa"]
    for technique in techniques:
      row['sim_ques_ans_how'+technique] = df_consistency_filtered['sim_ques_ans'+technique].values[0]
      row['sim_ques_how_why'+technique] = df_consistency_filtered['sim_ques_why'+technique].values[0]
      row['sim_ques_how_really'+technique] = df_consistency_filtered['sim_ques_really'+technique].values[0]

      row['sim_ans_base_how'+technique] = df_consistency_filtered['sim_ans_base'+technique].values[0]
      row['sim_ans_how_why'+technique] = df_consistency_filtered['sim_ans_why'+technique].values[0]
      row['sim_ans_how_really'+technique] = df_consistency_filtered['sim_ans_really'+technique].values[0]
      # row['correct_how'+technique] = df_consistency_filtered['label'+technique].values[0]

    question_approach = 'Challenge with really'

    # get the consistency score from the consistency dataframe with id, keyword and approach
    df_consistency_filtered = df_input[(df_input['id'] == id) & (df_input['question_keyword'] == question_keyword) & (df_input['question_approach'] == question_approach)]

    for technique in techniques:
      row['sim_ques_ans_really'+technique] = df_consistency_filtered['sim_ques_ans'+technique].values[0]
      row['sim_ques_why_really'+technique] = df_consistency_filtered['sim_ques_why'+technique].values[0]
      row['sim_ans_base_really'+technique] = df_consistency_filtered['sim_ans_base'+technique].values[0]
      row['sim_ans_why_really'+technique] = df_consistency_filtered['sim_ans_why'+technique].values[0]
      # row['correct_really'+technique] = df_consistency_filtered['label'+technique].values[0]

    question_approach = 'Challenge with why'

    # get the consistency score from the consistency dataframe with id, keyword and approach
    df_consistency_filtered = df_input[(df_input['id'] == id) & (df_input['question_keyword'] == question_keyword) & (df_input['question_approach'] == question_approach)]

    for technique in techniques:
      row['sim_ques_ans_why'+technique] = df_consistency_filtered['sim_ques_ans'+technique].values[0]
      row['sim_ans_base_why'+technique] = df_consistency_filtered['sim_ans_base'+technique].values[0]
      # row['correct_why'+technique] = df_consistency_filtered['label'+technique].values[0]


    # put the row back to the dataframe
    df_output.iloc[index] = row

    print('Done with index: ', index)
    
# save the dataframe
df_output.to_csv(output_file, index=False)

print('Done!'+output_file)




    



