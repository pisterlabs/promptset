from botfunctions import functionDefinitions
from coderun import runCode
import json
import openai

response = False
prompt_tokens = 0
completion_tokes = 0
total_tokens_used = 0
cost_of_response = 0

openai.api_key ="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"

def qagent(botinput):
    sysprompt = open('src/qabot.txt', 'r').read()
    messages = [{"role": "system", "content": sysprompt},{"role":"user","content":botinput}]
    print('qabot checking', messages)
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        temperature=0.5,
        top_p=0.5,
        messages=messages
    )
    print(response)
    return response
