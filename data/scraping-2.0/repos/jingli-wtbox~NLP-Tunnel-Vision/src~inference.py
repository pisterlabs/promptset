from openai import OpenAI
import os
from utils import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    RequestBody
)
import copy


def preproccess_request(request_body):
    user_prompt = copy.deepcopy(USER_PROMPT)
    for idx, h in enumerate(request_body.history):
        article, comment = h
        user_prompt += f"\n##\nArticle {idx}: " + article    + \
                        f"\nArticle {idx} Comment: " + comment
    user_prompt += "\n##\nNew Article: " + request_body.new_article + \
                    "\nNew Article Comment:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def infer(openai_client, request_body):
    res = ""
    try:   
        if not isinstance(request_body, RequestBody):
            res = "Request body must be a RequestBody object."
        elif not isinstance(request_body.history, list) or any(len(h)!=2 for h in request_body.history):
            res = "Payload must be a list of pairs of article and comment."
        elif len(request_body.history) == 0:
            res = "No article and comment pairs in payload."
        elif request_body.new_article.strip() is None or request_body.new_article.strip() == "":
            res = "New article must be provided."

        else:
            # convert request to openai messages
            messages = preproccess_request(request_body)

            response = openai_client.chat.completions.create(
                model = os.getenv("OPENAI_FINE_TUNED_MODEL_ID", None),
                messages = messages, 
                temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.9)),
            )
            res = response.choices[0].message.content
        
    except Exception as e:
        res = str(e)

    return res



if __name__ == "__main__":
    # set the environment variables before running this script
    os.environ['OPENAI_API_KEY'] = "sk-xxx"
    os.environ['OPENAI_FINE_TUNED_MODEL_ID'] = "xxx"
    os.environ['OPENAI_TEMPERATURE'] = "0.9"

    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    request_body = RequestBody(
        history = [
            ["This is first test article.", "this is a test comment."],
            ["This is secondary test article", "this is a secondary test comment."],
        ],
        new_article = "This is a new article."
    )

    res = infer(openai_client, request_body)
    print('New comment: \n', res)

