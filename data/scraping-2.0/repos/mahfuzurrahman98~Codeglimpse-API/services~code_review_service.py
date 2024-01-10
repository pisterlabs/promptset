import openai
from os import environ
from fastapi import FastAPI, HTTPException

# Load your API key from an environment variable or secret management service
openai.api_key = environ.get('OPENAI_API_KEY')


def get_response_openai(source_code, language):
    try:
        instructions = 'Do an in-depth code review, and improve comments, no additionl documentaion after or bvefore the code, just rewrite the code precisely.'

        messages = [
            {'role': 'system', 'content': 'You are an experienced software engineer reviewing a random code snippet.'},
            {'role': 'system', 'content': f'Code snippet is in {language}.'},
            {'role': 'user', 'content': source_code},
            {'role': 'user', 'content': instructions}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=messages,
            stream=True
        )
        # print(response)
    except Exception as e:
        # print("Error in creating campaigns from OpenAI:", str(e))
        raise HTTPException(503, detail=str(e))

    for chunk in response:
        choice_delta = chunk["choices"][0]["delta"]
        current_content = choice_delta.get("content", "")
        yield current_content
