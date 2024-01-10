import os
import openai
openai.api_key = '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'
prompt = ""; completion = ""
def askGPT(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
    {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content

while True:
    prompt = input()
    gptResponse = askGPT(prompt)
    print(gptResponse)