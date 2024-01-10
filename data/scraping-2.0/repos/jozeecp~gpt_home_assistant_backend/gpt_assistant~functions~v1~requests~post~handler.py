import os

import openai
from flask import request
from functions.v1.requests.post.schema.body import RequestBody
from libs.llm_utils import GPTInterface


def handler(request):
    # Get the request body
    request_body = RequestBody(**request.get_json())

    result = GPTInterface().execute_request(request_body.request_description)

    return {
        "statusCode": 200,
        "body": result,
    }
