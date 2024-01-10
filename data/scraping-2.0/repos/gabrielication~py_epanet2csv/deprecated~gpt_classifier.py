# import openai
# openai.organization = "org-eFRFvSQ3oY0Nbb1S287Z7F1Y"
# openai.api_key = "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"
# openai.Model.list()

import numpy as np
import pandas as pd
import json

def count_sim_duration_from_dataframe(df):
    duration = int(df['hour'].iloc[-1].split(":")[0])
    return duration

def convert_dataset_to_json(folder_path, filename):
    complete_file_path = folder_path+filename

    df = pd.read_csv(complete_file_path)

    sim_duration = count_sim_duration_from_dataframe(df)

    requests = []

    for hour in range(sim_duration):
        timestamp = str(hour)+":00:00"

        prompt = "Given this dataset of a smart water distribution network with these hydraulic values, can you spot leakages?\n\n"

        temp = df.loc[df['hour'] == timestamp]
        features = temp[['nodeID', 'base_demand', 'demand_value', 'head_value', 'pressure_value']]
        labels = temp['has_leak']

        features_to_string = features.to_string(index=False)

        prompt += features_to_string

        nodes_with_leaks = temp.loc[temp['has_leak'] == True]
        nodes_with_leaks = nodes_with_leaks['nodeID']

        if(len(nodes_with_leaks)>0):
            completion = "Yes there is a leak on node(s):\n"
            labels_to_string = nodes_with_leaks.to_string(index=False, header=False)
            completion += labels_to_string
        else:
            completion = "No, there is no leak in here."

        prompt_separator = "\n\n###\n\n"
        json_request = {"prompt":prompt+prompt_separator,"completion":completion}

        requests.append(json_request)

    print(len(requests))

    with open("my_file.json", "w") as f:
        for req in requests:
            json.dump(req,f)
            f.write('\n')


if __name__ == "__main__":
    # We read our entire dataset

    folder_path = "../tensorflow_datasets/one_res_small/gabriele_marzo_2023/"
    #  filename = "1M_one_res_small_no_leaks_rand_bd_filtered_merged.csv"
    filename = "1M_one_res_small_fixed_leaks_rand_bd_filtered_merged.csv"

    convert_dataset_to_json(folder_path,filename)