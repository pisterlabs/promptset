import openai
#from api_keys import OPENAI_API_KEY

def generate_chatgpt_response(history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=500,
        temperature=0.5,
        messages=history,
    )
    return response.choices[0].message.content

def generate_chatgpt_response_no_limit(history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        messages=history,
    )
    return response.choices[0].message.content

def generate_chatgpt4_response(history):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.5,
        messages=history,
    )
    return response.choices[0].message.content