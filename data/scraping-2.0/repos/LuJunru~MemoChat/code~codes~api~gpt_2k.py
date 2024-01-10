import re
import json
import openai
import time
import sys
import tiktoken

input_data = sys.argv[1]
openai_modelid = sys.argv[2]
openai.api_key = sys.argv[3]
output_path = sys.argv[4]
prompt_path = sys.argv[5]
encoding = tiktoken.encoding_for_model(openai_modelid)

q_pre = ""
qa_link = ""
MaxLen = 2048
TarLen = 512
TaskTarLen = {
    "chatting_dialogsum": MaxLen,
    "chatting_alpacagpt4": MaxLen,
    "writing_topiocqa": TarLen // 2,
    "writing_dialogsum": TarLen,
    "retrieval_dialogsum": 32,
    "retrieval_topiocqa": 32
}

prompts = json.load(open(prompt_path, "r"))

def normalize_chatting_outputs(model_outputs):
    def white_space_fix(text):
        lines = text.split("\n")
        result = []
        for line in lines:
            result.append(' '.join(line.split()))
        output = '\n'.join(result)
        return output
    return white_space_fix(model_outputs)

def gen_model_output(input_qs, task_type):
    input_qs_token_l = len(encoding.encode(input_qs))  # token num
    input_qs_word_l = len(input_qs.split(" "))  # word num
    qs_w_t_ratio = input_qs_word_l / input_qs_token_l
    max_word_num = int((MaxLen - TarLen) * qs_w_t_ratio)
    input_qs = " ".join(input_qs.split(" ")[-max_word_num:])
    target_len = TaskTarLen[task_type]
    messages = [{"role": "system", "content": input_qs}]
    for _ in range(5):
        try:
            chat = openai.ChatCompletion.create(
                model=openai_modelid, messages=messages, max_tokens=target_len, temperature=0.2
            )
            break
        except:
            time.sleep(5)
    model_outputs = chat.choices[0].message.content
    return model_outputs

def run_eval():
    data = json.load(open(input_data, "r"))
    output_data = []
    for d in data:
        print("=" * 20 + "start of question {}".format(d["id"]) + "=" * 20)
        new_d = d

        history = []
        for l_i in range(len(new_d["conversations"])):
            if l_i % 2 == 1:
                bot_thinking = {"retrieval": "", "summarization": ""}
                print("=" * 20 + "start of turn {}".format(l_i // 2 + 1) + "=" * 20)
                user = "user: " + new_d["conversations"][l_i - 1]["value"]

                system_insturction = prompts["chatting"]["system"]
                task_instruction = prompts["chatting"]["instruction"]
                task_case = "```\nRecent Dialogs:\n" + " ### ".join([hrd.replace("\n", " ") for hrd in history]) + "\n```\n\nUser Input:\n" + user + " ### bot: "
                qs = system_insturction + task_case + task_instruction
                print(qs + "\n\n")
                outputs = gen_model_output(qs, "chatting_dialogsum")
                outputs = normalize_chatting_outputs(outputs)
                history += [user, "bot: " + outputs]
                print("bot: " + outputs + "\n")
                print("=" * 20 + "end of turn {}".format(l_i // 2 + 1) + "=" * 20)
                new_d["conversations"][l_i]["thinking"] = json.dumps(bot_thinking)
                new_d["conversations"][l_i]["value"] = outputs

        output_data.append(new_d)
    json.dump(output_data, open(output_path, "w"), indent=2)

if __name__ == "__main__":
    run_eval()
