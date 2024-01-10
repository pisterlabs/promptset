'''
Code to fine tune our model with the jsonl file

Maverick Reynolds
10/20/2023
MangoAI

'''

import openai

def get_response():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )

    return response['choices'][0]['message']['content']

openai.File.create(
  file=open("results.jsonl", "rb"),
  purpose='fine-tune'
)

openai.File.list()

openai.FineTuningJob.create(training_file="file-H4nqeCUqMjnE2r9s8Jw6vM0d", model="gpt-3.5-turbo")

# Retrieve the state of a fine-tune
openai.FineTuningJob.retrieve("ftjob-n0XsbUxI6it7ullUmxlNH4a8")