import os
import openai
import pandas as pd
openai.api_key = os.environ["OPENAI_API_KEY"]

instruction = "Your task is to shorten the answer that is provided to you to max. 10 words."

df = pd.read_csv('data_merged_shortened_answer.csv', sep=';')


for index, row in df[639:].iterrows():
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
        "role": "system",
        "content": "Your task is to shorten inputs."
        },
        {
        "role": "user",
        "content": instruction + "\nAnswer: "+row.model_answer_uk
        }
    ],
    temperature=1,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    df.at[index,'shortened_uk'] = response['choices'][0]['message']['content']
    df.to_csv('data_merged_shortened_answer.csv', sep=';')


