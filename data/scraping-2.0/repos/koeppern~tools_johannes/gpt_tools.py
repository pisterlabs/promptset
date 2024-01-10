# gpt_tools.px
# 2023-10-08
# Functions like gpt_quick_response to ask openai gpt quickly a question
from dotenv import find_dotenv, load_dotenv, dotenv_values
import openai


def gpt_quick_response(message, system_message="", model="gpt-3.5-turbo", temperature=0.7):
    load_dotenv(find_dotenv(), override=True)

    openai.api_key = dotenv_values()["OPENAI_API_KEY"]

    messages = [
        {"role": "system", "content":system_message},
    ]

    if isinstance(message, str):
        messages.append(
            {"role": "user", "content": message}
        )        
    elif isinstance(message, list):
        for message in message:
            messages.append(
                {"role": "user", "content": message}
            )  
    else:
        messages = []
        
        print("message is neither a string nor a list")


    if len(system_message) == 0:
        messages.pop(0)

    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=messages
    )

    return response.choices[0].message.content