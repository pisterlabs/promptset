import openai
from revChatGPT.V3 import Chatbot

gpt_key = "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"
# openai.api_key = "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"


def search(input):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=input
        + "Given the following prompt, translate it into Python code\n\n {input} \n\n###",
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=1.0,
        presence_penalty=0.0,
        stop=['"""'],
    )
    return response.choices[0].text


def searchCustom(input):
    chatbot = Chatbot(api_key=gpt_key)
    response = chatbot.ask(
        f"Given the following prompt, translate it into Python code\n\n{input}",
    )
    return response

def ntrCustom(input):
    chatbot = Chatbot(api_key=gpt_key)
    response = chatbot.ask(
        f"Given the following Python code, translate it into English\n\n{input}",
    )
    return response

def debugCustom(input):
    chatbot = Chatbot(api_key=gpt_key)
    response = chatbot.ask(
        f"Given the following prompt, fix bugs in the below function. cite stackoverflow links and code to do this\n\n{input}",
    )
    return response

def debug(input):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"##### Fix bugs in the below function\n\n### Buggy Python\n {input} \n\n###",
        temperature=0.1,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"],
    )
    return response.choices[0].text


def explain(input):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=input
        + '"""\nHere\'s what the above class is doing, explained in a detailed way:\n1.',
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=1.0,
        presence_penalty=0.0,
        stop=['"""'],
    )
    return response.choices[0].text
