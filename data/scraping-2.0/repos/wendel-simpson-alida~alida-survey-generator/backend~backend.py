from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
import openai
import json

load_dotenv(".env")

app = Flask(__name__)
CORS(app)


@app.route('/get_params')
def get_params():
    filters = request.args.get('filters')
    return 'params'


# GET requests will be blocked
@app.route('/generate_survey', methods=['POST'])
def generate_survey():
    survey_items = []
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    request_data = request.get_json()
    model_input_context_text = 'Every survey question might have a type of Multiple Choice, Single Choice, Open Ended and output must be a json only. JSON format should be like {survey_title, questions[{question_name, question, question_type, options[]}]} but options field is not necessary for open ended questions.'
    if 'modelInputContext' in request_data:
        model_input_context_text = request_data.get("modelInputContext")

    if request_data.get('inputType') == 'quickStart':
        model_input = f'''{model_input_context_text} Generate a {request_data.get("numberOfQuestions")} question survey on {request_data.get("surveyType")} for company {request_data.get("companyName")} which is in the {request_data.get("industry")} industry'''

        model_output = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": model_input}]
        )

    if request_data.get("inputType") == 'textInput':
        model_input = f'''
                {model_input_context_text} Generate a survey 
                for company {request_data.get("companyName")} 
                which is in the {request_data.get("industry")} industry.
                Referring to the following information: {request_data.get('textInput')}'''

        model_output = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": model_input}]
        )

    content = json.loads(model_output.choices[0].message["content"])

    for question in content.get("questions"):
        if question["question_type"] == "Single Choice":
            dimension_member_items = []
            for option in question["options"]:
                dimension_member_items.append({
                    "properties": {
                        "html": option
                    }
                })
            survey_items.append({
                            "baseType": "SingleChoice",
                            "respondingPluginName": "SingleChoicePlugin",
                            "dimensions": {
                                "items": [
                                    {
                                        "dimensionMembers": {
                                            "items": dimension_member_items
                                        }
                                    }
                                ]
                            },
                            "properties": {
                                "name": question["question"],
                                "html": "<p>"+question["question"]+"</p>"
                            }
                        })

        if question["question_type"] == "Multiple Choice":
            dimension_member_items = []
            for option in question["options"]:
                dimension_member_items.append({
                                                    "properties": {
                                                        "html": option
                                                    }
                                                })
            survey_items.append({
                            "baseType": "MultipleChoice",
                            "respondingPluginName": "MultipleChoicePlugin",
                            "dimensions": {
                                "items": [
                                    {
                                        "dimensionMembers": {
                                            "items": dimension_member_items
                                        },
                                        "properties": {
                                            "maximumSelection": 0,
                                            "minimumSelection": 1
                                        }
                                    }
                                ]
                            },
                            "properties": {
                                "name": question["question"],
                                "html": "<p>"+question["question"]+"</p>"
                            }
                        })

        if question["question_type"] == "Open Ended":
            survey_items.append({
                "baseType": "OpenEnded",
                "respondingPluginName": "OpenEndNoValidationPlugin",
                "properties": {
                    "name": question["question"],
                    "html": "<p>"+question["question"]+"</p>"
                }
            })

    survey = {
        "defaultLocale": "en-CA",
        "steps": {
            "items": [
                {
                    "baseType": "Block",
                    "respondingPluginName": "PageBreakPlugin",
                    "steps": {
                        "items": survey_items
                    }
                }
             ]
        }
    }

    return survey


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
