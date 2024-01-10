import os
import openai
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('data_global_opinions_us_uk.csv', sep=';')


for index, row in df[501:].iterrows():
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "You are a supposed to determine the type of a question. Please provide high quality type determination. Use the provided question as well as the answer options to determine it. Provide only the question type as response."
        },
        {
        "role": "assistant",
        "content": "Please only categorize the question as one of the following types: 1) Numerical Scale 2) Multiple Choice 3) Likert Scale 4) Ordinal Scale 5) Binary Choice"
        },
        {
        "role": "user",
        "content": "Question: " + str(row.question) + " Answer options: " +str(row.options)
        }
    ],
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    df.at[index,'question type'] = response['choices'][0]['message']['content']
    df.to_csv('data_global_opinions_us_uk.csv', sep=';')


