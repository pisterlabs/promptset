import os
import openai
import json
import time

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

ppo_dir = "../ppo/"
dataset2_path = "../sft/line_human-touched.json"
result_path = "../sft/sft_dataset.json"

with open(dataset2_path, 'r') as json_file:
    old_data = json.load(json_file)

for filename in os.listdir(ppo_dir):
    f = os.path.join(ppo_dir, filename)
    with open(f, 'r') as json_file:
        new_data = json.load(json_file)

    with open(result_path, 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    with open(result_path, 'w', encoding='utf-8') as json_file:
        print(len(new_data))
        for i in range(len(new_data)):
            if new_data[i]["instruction"].find("<수정된 데이터>") != -1 and new_data[i]["instruction"].find("for row") == -1:
                data = {
                    "instruction" : new_data[i]["instruction"].replace("<수정된 데이터>", "<데이터>"),
                    "input" : "",
                    "output" : old_data[i%500]["output"],
                    "target" : new_data[i]["target"],
                    "category" : new_data[i]["category"],
                    "log_id" : new_data[i]["log_id"]
                }
                dataset.append(data)
        json.dump(dataset, json_file, indent = 4, ensure_ascii=False)