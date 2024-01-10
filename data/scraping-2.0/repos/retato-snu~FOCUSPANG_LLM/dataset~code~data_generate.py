import os
import openai
import json
import time

from dotenv import load_dotenv

load_dotenv()
dataset_path = os.environ.get('SFT_DATASET_PATH')
result_path = os.environ.get('PPO_DATASET_PATH')

openai.api_key = os.environ.get('OPENAI_API_KEY')

impulsiveness_student_data = []
impulsiveness_count_data = []
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

def categorize():
    with open(dataset_path, 'r') as json_file:
        dataset = json.load(json_file)
    for data in dataset:
        category = data["category"]
        if category == "impulsiveness-count":
            impulsiveness_count_data.append(data)
        else:
            impulsiveness_student_data.append(data)

def generate_is(n):
    for data in impulsiveness_student_data:
        print(1)
        ins = data["instruction"]
        prefix = "너는 주어진 데이터를 요청에 맞춰 수정하고, 그 수정한 데이터를 출력하는 AI이다.\n\n"
        suffix = "```\n\n<요청>\n```\n각 데이터의 점수를 그것에 '-0.05보다 크고 0.05보다 작은 임의의 수'를 더한 값으로 바꿔줘. 단 40보다 큰 수는 여전히 40보다 큰 수로 바꿔져야해. 그리고 40보다 작은 수를 바꾼 값도 여전히 40보다 작아야 해.\n```\n수정된 데이터의 출력 형식은 다음과 같아.\n<수정된 데이터>\n```\n```\n"
        message = [
            {"role":"user", "content":(prefix+get_data(ins)+get_data_description(ins)+suffix)},
        ]
        for i in range(n):
            response = request_gpt(message)
            idx1 = response.find("<수정된 데이터>")
            idx2 = find_2nd(response, "```")
            res = ins.replace(get_data(ins),response[idx1:idx2+3]+"\n")
            r_data = {
                "instruction": res,
                "input": "",
                "output": data["output"],
                "target": data["target"],
                "category": data["category"],
                "log_id": data["log_id"]
            }
            print(res)
            result_data.append(r_data)
        

def generate_ic(n):
    for data in impulsiveness_count_data:
        print(2)
        ins = data["instruction"]
        prefix = "너는 주어진 데이터를 요청에 맞춰 수정하고, 그 수정한 데이터를 출력하는 AI이다.\n\n"
        suffix = "```\n\n<요청>\n```\n각 데이터의 '충동 성향이 있는 학생 수'를 그것에 '-0.05보다 크고 0.05보다 작은 임의의 수'를 더한 값으로 바꿔줘. 단 40보다 큰 수는 여전히 40보다 큰 수로 바꿔져야해. 그리고 40보다 작은 수를 바꾼 값도 여전히 40보다 작아야 해.\n```\n수정된 데이터의 출력 형식은 다음과 같아.\n<수정된 데이터>\n```\n```\n"
        message = [
            {"role":"user", "content":(prefix+get_data(ins)+get_data_description(ins)+suffix)},
        ]
        for i in range(n):
            response = request_gpt(message)
            idx1 = response.find("<수정된 데이터>")
            idx2 = find_2nd(response, "```")
            res = ins.replace(get_data(ins),response[idx1:idx2+3]+"\n")
            r_data = {
                "instruction": res,
                "input": "",
                "output": data["output"],
                "target": data["target"],
                "category": data["category"],
                "log_id": data["log_id"]
            }
            print(res)
            result_data.append(r_data)

if __name__ == "__main__":
    n = int(input("만들 데이터 개수(500의 배수로)"))
    n = int(n / 500)
    categorize()
    generate_is(n)
    generate_ic(n)
    with open(result_path, "w", encoding='utf-8') as json_file:
        json.dump(result_data, json_file, indent = 4, ensure_ascii=False)