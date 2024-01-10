import os
import openai
import json
import time

from dotenv import load_dotenv

load_dotenv()
dataset_path = '../sft/sft_dataset1.json'
result_path = './rm_dataset.json'

openai.api_key = os.environ.get('OPENAI_API_KEY')

result_data = []

def request_gpt(message):
    while(True):
        try:
            response = openai.ChatCompletion.create(
                model = 'gpt-3.5-turbo',
                messages = message
            )['choices'][0]['message']['content']
            break
        except openai.error.Timeout as e:
            print("rate limit/waiting")
            time.sleep(10)
            continue
        except openai.error.APIError as e:
            print("502")
            time.sleep(10)
            continue
    return response

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

def get_data(message):
    idx1 = message.find("<데이터>")
    idx2 = message.find("첫번째")
    return message[idx1:idx2]

def get_data_description(message):
    idx1 = message.find("첫번째")
    idx2 = message.find("<데이터>를 이용해")
    return message[idx1:idx2]


def generate():
    with open(dataset_path, "r", encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    for data in dataset:
        ins = data["output"]
        s = "아래의 분석 결과의 문장의 어투를 거칠게 해줘. '~다', '~했어' 등 항상 반말로 문장이 끝나야 해. <분석 결과> {sum}".format(sum = ins)
        message = [
            {"role":"user", "content": s},
        ]
        response = request_gpt(message)
        r_data = {
            "instruction": data["instruction"],
            "completion_a": data["output"],
            "completion_b": response
        }
        print(response)
        result_data.append(r_data)


if __name__ == "__main__":
    generate()
    with open(result_path, "w", encoding='utf-8') as json_file:
        json.dump(result_data, json_file, indent = 4, ensure_ascii=False)