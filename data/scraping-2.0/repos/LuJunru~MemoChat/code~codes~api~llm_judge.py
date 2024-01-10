import sys
import os
import json
import re
import openai
import time
import tiktoken

input_data = sys.argv[1]
openai_modelid = sys.argv[2]
openai.api_key = sys.argv[3]
output_path = sys.argv[4]
prompt_path = sys.argv[5]
encoding = tiktoken.encoding_for_model(openai_modelid)

prompts = json.load(open(prompt_path, "r"))
judge_prompt_raw = prompts["judge"]["system"]

def gen_model_output(input_qs):
    input_qs_token_l = len(encoding.encode(input_qs))  # token num
    input_qs_word_l = len(input_qs.split(" "))  # word num
    qs_w_t_ratio = input_qs_word_l / input_qs_token_l
    max_word_num = int(4096 * qs_w_t_ratio)
    input_qs = " ".join(input_qs.split(" ")[-max_word_num:])
    messages = [{"role": "system", "content": input_qs}]
    chat = None
    for _ in range(5):
        try:
            chat = openai.ChatCompletion.create(
                model=openai_modelid, messages=messages
            )
            break
        except:
            time.sleep(5)
    if chat is None:
        return "Cannot generate output."
    model_outputs = chat.choices[0].message.content
    return model_outputs

data = json.load(open(input_data, "r"))

# do llm judge
output_ratings = []
for d in data:
    print("=" * 20 + "Processing: " + d["id"] + "=" * 20)
    judge_conversation = []
    d_conversations = d['conversations']
    last_q = d_conversations[-2]
    turn_infos = last_q["turn-info"].split("-")
    r_turns = [turn_infos[0] + "-" + turn_infos[1]]
    if len(turn_infos) == 5:
        r_turns.append(turn_infos[2] + "-" + turn_infos[3])
    for l_i in range(len(d_conversations) // 2 - 1):
        if d_conversations[l_i * 2]["turn-info"][:-2] in r_turns:
            judge_conversation.append("user: " + d_conversations[l_i * 2]["value"])
            judge_conversation.append("bot: " + d_conversations[l_i * 2 + 1]["value"])
    judge_prompt = judge_prompt_raw.replace("RCH_0", "\n".join(judge_conversation)).replace("UQ_1", "user: " + last_q["value"]).replace("BR_2", "bot: " + d_conversations[-1]["value"])
    print(judge_prompt)
    print('-' * 20)
    outputs = gen_model_output(judge_prompt)
    print(outputs)
    print("=" * 20 + "Processed: " + d["id"] + "=" * 20)
    match = re.search(r'\[\[(\d+)\]\]', outputs)
    try:
        rating = int(match.group(1))
    except:
        rating = None
    output_ratings.append({
        "id": d["id"],
        "type": d["type"],
        "judge_prompt": judge_prompt,
        "evaluation": outputs,
        "rating": rating
    })
json.dump(output_ratings, open(output_path, "w"), indent=2)

# compute score
count = {
    "continuation": [],
    "retrospection": [],
    "conjunction": []
}
for d in output_ratings:
    if d["type"] == "continuation":
        count["continuation"].append(d["rating"])
    elif d["type"] == "retrospection":
        count["retrospection"].append(d["rating"])
    elif d["type"] == "conjunction":
        count["conjunction"].append(d["rating"])
print("Retrospection Score: {}, Continuation Score: {}, Conjunction Score: {}, Overall Score: {} of file {}".format(
    round(sum(count["retrospection"]) / len(count["retrospection"]), 2),
    round(sum(count["continuation"]) / len(count["continuation"]), 2),
    round(sum(count["conjunction"]) / len(count["conjunction"]), 2),
    round(sum(count["continuation"] + count["retrospection"] + count["conjunction"]) / len(count["continuation"] + count["retrospection"] + count["conjunction"]), 2),
    input_data
))
