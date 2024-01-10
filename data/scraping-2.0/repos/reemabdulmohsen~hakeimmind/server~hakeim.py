import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
import json


openai.organization = "org-yiFH362hXjqBaJ20VwGBQTqL"
openai.api_key = "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"


def process_model_output(output):
    try:
        json_output = json.loads(output)
        return json_output
    except:
        raise {"error":"not found"}


def generate_mind_map(disease):
    prompt = '''Generate a mind map in dictionary format for the disease: '''+ disease+'''.
    Please structure the mind map with the following format:
    {
    "Root": "{root_node}",
    "Children": [
    {child_1},
    {child_2},
    {child_3},
    ...
    ]
    }
    Ensure that each child node is a dictionary with a "name" key specifying the name of the node.
    Please generate the mind map accordingly.'''

    # Send the prompt to ChatGPT and receive the model's response
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": prompt},
        ]
    )

    # Process the model's output into a standardized format
    mind_map = process_model_output(completion.choices[0].message["content"])

    # Convert the mind map to a formatted string or JSON representation
    mind_map_str = json.dumps(mind_map, indent=2)  # Example: JSON representation

    return mind_map_str


def GenerateMindMap(diseases):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": "create a mindmap for" + diseases + " in json format"},
        ]
    )


    return completion.choices[0].message["content"]


# creating a Flask app
app = Flask(__name__)
CORS(app)


# first route "empty route"
@app.route('/', methods=["GET"])
def index():
    return "first route"


# second route "diseases"
@app.route('/diseases', methods=["GET"])
def diseases():
    diseases = str(request.args.get('diseasesName'))
    json_dump = generate_mind_map(diseases)
    response = json.dumps(json_dump)
    return response


if __name__ == "__main__":
    app.run(port=7776)

