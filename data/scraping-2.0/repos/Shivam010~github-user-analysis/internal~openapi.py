import os
import json
import openai

OPENAI_KEY = os.getenv("OPENAI_KEY")

systemPrompt = "You are very talented developer. You are provided some extracted details of all the files in a repository of a user (like what are the components or functions or comments, etc are there in them) in a particular format. Learn everything you can from the provided information. First the user will provide the details of the repository and then in the next message, a question will be provided for you to answer based on the provided information of the repository. \n > IMPORTANT NOTE: Always return a brief summary for the answer and if you are not able to find or deduce anything just respond with 'sorry, insufficient information!' \n"


def generate_response(repoDetail: str, query: str):
    if not OPENAI_KEY:
        return {
            "usage": {},
            "message": "error: OPENAI_KEY not provided",
        }

    openai.api_key = OPENAI_KEY
    messages = [
        {
            "role": "system",
            "content": systemPrompt,
        },
        {"role": "user", "content": repoDetail},
        {
            "role": "assistant",
            "content": "Ok! What's the question?",
        },
        {"role": "user", "content": query},
    ]
    # print(json.dumps(messages))

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    resp = []
    for cho in response["choices"]:
        if cho["message"] and cho["message"]["content"]:
            resp.append(cho["message"]["content"].strip())

    return {
        "message": "\n".join(resp),
        "usage": response["usage"],
    }
