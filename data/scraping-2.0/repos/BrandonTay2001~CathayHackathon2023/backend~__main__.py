import json
from pathlib import Path

import openai
from flask import Flask, request
from flask_cors import CORS

from backend.definitions import root_dir

app = Flask(__name__)
CORS(app)

openai.api_type = "azure"
openai.api_base = "https://team-21.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = "f9d497ddba0e4f698c4064664785cd54"

with open(Path(root_dir / "temp.json")) as f:
    backup_json = json.load(f)


@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    prompt = f"""
    You are a trip planner AI. 
    Given a user-given destination, start date and end date, prompt, pacing, generate a detailed itinerary in the form of a list of JSON objects with the following fields: name, startTime, endTime
    If prompt is "historical trip filled with delicious food", then the response should be in this form:
    [{{"name": "Kyoto Shrine", "reason": "The place is a historical heritage, which suits the requirement of this trip.", "startTime": "2023-12-11T09:00:00.000Z", "endTime": "2023-12-11T12:00:00.000Z"}}, {{"name": "Kyoto Katsugyu", "reason": "This is a popular restaurant that serves delicious fried pork.", "startTime": "2023-12-11T13:00:00.000Z", "endTime": "2023-12-11T14:30:00.000Z"}}]
    
    Name fields should be specific and real locations. For example, do not mention "breakfast at local cafe" but instead "Breakfast at Lai Chi Kok".
    Apart from attractions, also fit in lunch and dinner recommendations between activities. Durations of each activity should be between 30 minutes to 90 minutes.
    
    Request:
    Destination: {request.json["city"]}
    Start date: {request.json["startDate"]}
    End date: {request.json["endDate"]}
    Prompt: {request.json["prompt"]}
    Pacing: {request.json["pace"]}
    
    Response (do not include newline characters in response):
    """

    try:
        completion = openai.Completion.create(
            engine="generate-timetable",
            prompt=prompt,
            temperature=0.9,
            max_tokens=4000,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            stop=None
        )
    except Exception as ex:
        print(ex)
        return backup_json

    res_text = completion["choices"][0]["text"]
    print(res_text)
    try:
        res_json = json.loads(res_text)
    except Exception as ex:
        print(ex)
        return backup_json

    return res_json


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3002)
