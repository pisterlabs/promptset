import os
import openai
import json
import time

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')


directory = "./logs"
result_path = "./rm_dataset.json"
pathset = []
rm_dataset = []

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

def track_files(path):
    if os.path.isfile(path):
        if path.find("xlsx") == -1 and path.find("2_final_touch") != -1:
            pathset.append(path[:-17])
            print(path[:-17])
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            f = os.path.join(path, filename)
            track_files(f)

def extract_dataset():
    for directory in pathset:
        f1 = open(directory + "0_analyzation.txt", "r")
        analyzation = f1.read()
        f2 = open(directory + "2_final_touch.txt", "r")
        finaltouch = f2.read()
        f3 = open(directory + "1_filtering.txt", "r")
        filtering = f3.read()

        idx1 = analyzation.find("[Content]")
        idx2 = analyzation.find("=======================")
        instruction = analyzation[idx1 + 9 : idx2]

        idx1 = find_2nd(analyzation, "<분석 결과>")
        idx2 = find_2nd(analyzation, "=======================")
        bad_completion1 = analyzation[idx1+7:].replace("=", "")

        idx1 = find_2nd(filtering, "<재작성된 분석 결과>")
        idx2 = find_2nd(filtering,"=======================")
        bad_completion2 = filtering[idx1 + 13 : ].replace("=", "")

        idx1 = find_2nd(finaltouch, "[Role]")
        finaltouch = finaltouch[idx1:]
        idx1 = find_2nd(finaltouch, "[Role]")
        finaltouch = finaltouch[idx1:]
        idx1 = finaltouch.find("<최종 분석 결과>")
        idx2 = finaltouch.find("<최종 분석 결과에 대한 평가>")
        good_completion = finaltouch[idx1 + 11: idx2]

        rm_dataset.append({
            "instruction": instruction,
            "completion_0": bad_completion1,
            "completion_1": good_completion,
            "ranking" : [
                1,
                0
            ]
        })

        rm_dataset.append({
            "instruction": instruction,
            "completion_0": good_completion,
            "completion_1": bad_completion2,
            "ranking" : [
                0,
                1
            ]
        })

    with open(result_path, "w", encoding='utf-8') as json_file:
        json.dump(rm_dataset, json_file, indent = 4, ensure_ascii=False)


if __name__ == "__main__":
    track_files(directory)
    extract_dataset()