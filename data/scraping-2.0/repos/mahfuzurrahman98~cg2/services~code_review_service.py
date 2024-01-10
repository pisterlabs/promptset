import openai
from os import environ
import json

import requests

# Load your API key from an environment variable or secret management service
openai.api_key = environ.get('OPENAI_API_KEY')


def review_code(source_code):
    try:
        service_url = environ.get('CODE_REVIEW_SERVICE_URL')
        # send a post request to the code review service with the source_code, endpoint is /review-code
        payload = json.dumps({
            "source_code": source_code
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request(
            "POST",
            service_url + '/review-code',
            headers=headers,
            data=payload
        )

        # it will return a json with {message, and data}, just take the data and return it
        response_text = json.loads(response.text)
        # print(response_text)
        return response_text['data']

    except Exception as e:
        # print(e)
        raise e
