
import os
import openai

openai.api_key = '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'
result = openai.Model.list()
print(result)

prompt = 'how do you think about the weather today?'
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temprature=0.9,
    messages=[
        {"role": "system", "content": prompt},
    ]
)
reply = response.choices[0].message.content
print(reply)
