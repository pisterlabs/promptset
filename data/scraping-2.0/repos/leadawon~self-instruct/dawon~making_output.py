import json
import openai


def create(ins, dial):
    res=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role":"system","content": f"{dial}\n \
        {ins} of dialogue, answer as short as possible"}
        ]
    )
    return res["choices"][0]["message"]["content"]

print("start!")

with open('./apikey.json') as f:
        api_key = json.load(f)["apikey"]
openai.api_key = api_key

ins_list = []

with open("./instructions.jsonl", "r") as fin:
    for line in fin:
        instruction_info = json.loads(line)
        ins_list.append(instruction_info["instruction"])

print("instruction loaded && " + f"length : {len(ins_list)}")

with open("./dailydialogue.json","r") as json_file:
    dia_list = json.load(json_file)["list"]

print("dailydialogue loaded && " + f"length : {len(dia_list)}")

result_list = []
MIN_NUM = 10
for idx in range(min(len(dia_list), len(ins_list), MIN_NUM)):
    if idx < MIN_NUM:
        print(f"trial {idx+1}")
    else:
        print(f"...{idx+1}",end="")
    result_list.append({"instruction":ins_list[idx], "input":dia_list[idx], "output":create(ins_list[idx], dia_list[idx])})
    
with open("./output.jsonl","w") as fout:
    for data in result_list:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")