import openai
import os

openai.api_key = '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'

with open('jobDescription.txt', 'r') as file:
    jobDescription = file.read()
with open('resumeFormatted.txt', 'r') as file:
    resumeFormatted = file.read()

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
              messages=[{"role": "system", "content": jobDescription},
                        {"role": "user", "content": resumeFormatted}
              ])

print(response["choices"][0]["message"]["content"])
