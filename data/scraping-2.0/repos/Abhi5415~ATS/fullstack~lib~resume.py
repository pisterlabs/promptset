import openai
import json
import os
import lib.constants as constants

openai.api_key = os.getenv("STREAMLIT_OPENAI_KEY")


functions = [
    {
        "name": "rate_applicant",
        "description": "approve or reject an applicant based on their resume and portfolio data",
        "parameters": {
            "type": "object",
            "properties": {
                "approve": {
                    "type": "boolean",
                    "description": "whether or not to approve the applicant",
                },
                "reason": {
                    "type": "string",
                    "description": "why the applicant was approved or rejected",
                },
            },
            "required": ["approve"],
        },
    }
]


def get_prompt(resume, github_data, job_description):
    return f"""
This is an applicant:
{resume}

This is their GitHub profile:
{json.dumps(github_data)}

Use their github profile as a measure of their experience level.
This is a rough estimate of their experience level, but it's not perfect because some people have private repos.

Are they a good fit for the following job description?
{job_description}

You should be confident in your decision, but you don't need to be 100% certain.
Be highly selective and only approve applicants that you think are top candidates, if you think you can find a better fit, reject the applicant.
The resume should show evidence of producing high-value work, if the resume doesn't show this, reject the applicant.
Remember that most people lie or exaggerate their resume, if the resume seems suspect, reject the applicant.

Use the rate_applicant function to approve or reject the applicant and provide reasoning for your decision.
"""


def get_rating(resume, github_data):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "user",
                "content": get_prompt(resume, github_data, constants.JOB_DESCRIPTION),
            }
        ],
        functions=functions,
    )

    response_message = response["choices"][0]["message"]
    response = json.loads(response_message["function_call"]["arguments"])

    return response["approve"], response["reason"]
