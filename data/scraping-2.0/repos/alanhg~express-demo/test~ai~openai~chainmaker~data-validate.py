import json

import openai
import tiktoken # for token counting
import numpy as np
from collections import defaultdict
data_path = 'data/3.jsonl'

# Load the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)
# 上传训练文件
# training_response = openai.File.create(
#     file=open(data_path, "rb"), purpose="fine-tune"
# )
# training_file_id = training_response["id"]
# print("Training file ID:", training_file_id)

# --- end
#
#
# openai.FineTuningJob.create(training_file="file-RtavtDuKOOXVGKo1OQPzpajR", model="ft:gpt-3.5-turbo-0613:alan-self::84NyAlsJ")
# result=openai.FineTuningJob.list(limit=10)
# print(result)
#
jobstate=openai.FineTuningJob.retrieve("ftjob-ymeiMPMOHMg7y4vHAmkCx6b2")
print(jobstate)




