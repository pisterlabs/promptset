import openai
from dotenv import dotenv_values

config = dotenv_values(".env")

def gen_response(text, model, stop, tokens):
    openai.api_key = config["API_KEY"]
    response = openai.Completion.create(
    model=model,
    prompt=text,
    temperature=0.7,
    max_tokens=tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=stop
    )
    return response["choices"][0]["text"]