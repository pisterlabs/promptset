import os
from typing import List
import copy
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import json
import jsonlines

load_dotenv()
key = os.environ.get("OPENAI_KEY")
address = os.environ.get("MATCH_SERVICE_ADDRESS")
api_address = os.environ.get("BACKEND_REACH_ADDRESS")
openai.api_key = key
app = Flask(__name__)


class Cache:
    storage = {}


@app.route("/")
def index():
    return "hello world!"


@app.route("/api/matcher", methods=['GET', 'POST'])
def get_matching():
    content = request.form
    text = content['text']

    print(text)
    query_as_list = extract(text)
    query = str(query_as_list).replace("[", "").replace("]", "").replace("'", "")

    result = openai.Engine("davinci").search(
        file="file-cEbtzr5OPbMfgHXroCUm5hT4",
        query=query,
        return_metadata=True,
        max_rerank=10
    )

    values = []
    for value in result["data"]:
        values.append({"score": value["score"], "name": value["metadata"]})

    tmp_score = 0
    max_val = None
    for value in values:
        if value["score"] > tmp_score:
            tmp_score = value["score"]
            max_val = value["name"]

    print(max_val)

    resp = requests.get(api_address + "/diseases/by-name/" + max_val)
    disease_id = resp.json()['id']

    dep = requests.get(api_address + "/departments/by-disease/" + str(disease_id)).json()
    dep_name = dep['name']
    print(dep_name)

    response = {
        "text": "you are likely to have: " + max_val + "\nwe recommend going to the Department of " + dep_name + " first"
    }

    return jsonify(response)


@app.route("/api/cache/update")
def force_update():
    val = requests.get(api_address + "/e/diseases").json()
    update_cache(val)
    print(Cache.storage)
    return "Success"


@app.route("/api/cache")
def get_cache():
    return jsonify(Cache.storage)


def update_cache(val):
    Cache.storage.clear()
    Cache.storage = val
    regenerate_file()


def extract(text) -> List[str]:
    response = openai.Completion.create(
        engine="davinci",
        prompt="Read this patient phone call:\n\"\"\"\n {} \n\"\"\"\nAnswer the following questions:\n\n1. What is the patients name?\n2. What symptoms is he mentioning?\n3. Return the symptoms as a Python list object\n4. What disease could he have?\n5. How can it be treated?\n6. What kind of doctor should the patient see?\n7. Write \"Hello World\" on the console\n\nAnswers:\n1.".format(
            text),
        temperature=0.3,
        max_tokens=59,
        top_p=0.3,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["7."]
    )

    text_output = response["choices"][0]["text"]
    b = "234567."
    for char in b:
        text_output = text_output.replace(char, "")

    result = text_output.splitlines()
    return result[2]


def regenerate_file():
    temp = []

    for val in Cache.storage:
        symptoms = ""
        metadata = val["name"]

        for symptom in val["symptoms"]:
            symptoms += symptom["name"] + ", "
        size = len(symptoms)
        mod_string = symptoms[:size - 2]
        symptoms = mod_string

        temp.append({'text': symptoms, "metadata": metadata})

    with open('services/output.jsonl', 'w') as outfile:
        for entry in temp:
            json.dump(entry, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    host = address.split(':')[0]
    port = address.split(':')[1]
    app.run(host=host, port=port, debug=True)
