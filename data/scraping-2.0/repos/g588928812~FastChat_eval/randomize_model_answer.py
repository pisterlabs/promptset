import argparse
import json
import os
import time

import requests
import openai
import tqdm
import re

import shortuuid
import logging

from datetime import datetime

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line, strict=False))
        return json_list

def import_json(file_path):
    file_path = os.path.expanduser(file_path)
    f = open(file_path)
    data = json.load(f)
    f.close()

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-a", "--answer-file")
    parser.add_argument("-o", "--output-answer-file")
    args = parser.parse_args()

    assert(args.answer_file and args.output_answer_file)

    answer_jsons = get_json_list(args.answer_file)

    for answer in answer_jsons:
        answer["answer_id"]=shortuuid.uuid()
        answer["model_id"]=answer["model_id"]+"-shifted"
        answer["question_id"]=(answer["question_id"]+1) if (answer["question_id"]+1) <= len(answer_jsons) else 1

    with open(os.path.expanduser(args.output_answer_file), "w") as ans_file:
        for line in answer_jsons:
            ans_file.write(json.dumps(line) + "\n")

