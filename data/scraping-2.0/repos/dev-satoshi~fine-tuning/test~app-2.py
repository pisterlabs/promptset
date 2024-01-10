import json

import openai
from datasets import load_dataset

api_key = "F8u70hM6PUFmPnPzoFHUT3BlbkFJnr2R4OFHT5O2Vd2Hn4u5"

# 独自のデータを読み込む
dataset = load_dataset("bbz662bbz/databricks-dolly-15k-ja-gozarinnemon")
dataset["train"][0]["instruction"]
dataset["train"][0]["output"]


# instructionは指示だったり質問のこと
# outputがinstructionの回答のこと

list_message = []
num_data = 10

for i in range(num_data):
    instruction = dataset["train"][i]["instruction"]
    output = dataset["train"][i]["output"]
    message = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    list_message.append(message)

with open("output.json", "w") as file:
    for messages in list_message:
        json_line = json.dumps({"messages": messages})
        file.write(json_line + "\n")
