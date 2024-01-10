import os
import openai

from dotenv import load_dotenv


load_dotenv()

# OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT-4 call
def gpt_4_answer(
    messages,
    model="gpt-4",
    max_tokens=750,
    temperature=0.6,
    top_p=0.9,
    frequency_penalty=1.2,
    presence_penalty=0.5,
):
    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty, 
        "max_tokens": max_tokens,
    }

    response = openai.ChatCompletion.create(**completion_params)

    return response["choices"][0]["message"]["content"]


# GPT-3.5 turbo 16k call
def gpt_3_5_turbo_16k_answer(
    messages,
    model="gpt-3.5-turbo-16k",
    max_tokens=750,
    temperature=0.6,
    top_p=0.9,
    frequency_penalty=1.2,
    presence_penalty=0.5,
):
    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty, 
        "max_tokens": max_tokens,
    }

    response = openai.ChatCompletion.create(**completion_params)

    return response["choices"][0]["message"]["content"]