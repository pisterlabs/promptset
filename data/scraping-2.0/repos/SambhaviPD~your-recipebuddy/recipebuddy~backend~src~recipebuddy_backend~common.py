import openai

from .configuration import ResponseModel

def get_recipe_using_openai(api_key, prompt, success_message):
    openai.api_key = api_key
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=2048,
        temperature=0,
    )
    final_response = ResponseModel(
        success=True, \
        message=success_message, \
        data=response
    )
    return final_response