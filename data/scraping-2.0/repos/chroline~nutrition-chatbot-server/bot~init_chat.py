import openai


def init_chat(api_key, messages):
    response = openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-3.5-turbo",
        stream=True,
        messages=messages
    )
    for chunk in response:
        yield chunk['choices'][0]['delta']
