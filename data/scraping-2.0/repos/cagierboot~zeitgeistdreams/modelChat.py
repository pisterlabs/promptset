

import openai

openai.api_key = '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'

response = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo-0613:markortega::8Ev0Aizf",
    messages=[
        {"role": "system", "content": "You are an assistant that is knowledgable about Reddit news trends and can perform analyses on trends"},
        {"role": "user", "content": "What are some common trends recently"}
    ]
)

print(response.choices[0].message['content'])

