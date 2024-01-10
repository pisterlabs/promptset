# This script to fine tune GPT3 model on movie scripts
import requests
import os
import openai
from pprint import pprint

# set openai API key
openai.api_key = os.getenv("OPENAI_KEY")
open_ai_api_key = os.getenv("OPENAI_KEY")


# function to upload file to openai
def file_upload(file_name, purpose="fine-tuning"):
    # open file
    with open(file_name, "r", encoding="utf-8") as f:
        # read file
        file = f.read()
    # upload file
    response = openai.File.create(file=file, purpose=purpose)
    # print response
    pprint(response)
    # return file id
    return response


def file_list():
    # list files
    response = openai.File.list()
    # print response
    pprint(response)
    # return response
    return response


# function to fine tune openai model


def finetune_model(fileid, suffix, model="davinci"):
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % open_ai_api_key,
    }
    payload = {"training_file": fileid, "model": model, "suffix": suffix}
    resp = requests.request(
        method="POST",
        url="https://api.openai.com/v1/fine-tunes",
        json=payload,
        headers=header,
        timeout=45,
    )
    pprint(resp.json())


def finetune_list():
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % open_ai_api_key,
    }
    resp = requests.request(
        method="GET",
        url="https://api.openai.com/v1/fine-tunes",
        headers=header,
        timeout=45,
    )
    pprint(resp.json())


def finetune_events(ftid):
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % open_ai_api_key,
    }
    resp = requests.request(
        method="GET",
        url="https://api.openai.com/v1/fine-tunes/%s/events" % ftid,
        headers=header,
        timeout=45,
    )
    pprint(resp.json())


def finetune_get(ftid):
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % open_ai_api_key,
    }
    resp = requests.request(
        method="GET",
        url="https://api.openai.com/v1/fine-tunes/%s" % ftid,
        headers=header,
        timeout=45,
    )
    pprint(resp.json())


if __name__ == "__main__":
    # upload file
    response = file_upload("movieScriptGenerator.jsonl")
    file_id = response["id"]

    # list files
    file_list()
    finetune_model(file_id, "scripts")
    finetune_list()
