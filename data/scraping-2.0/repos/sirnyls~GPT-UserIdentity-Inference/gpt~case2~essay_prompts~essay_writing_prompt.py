import os
import openai
import pandas as pd
openai.api_key = os.environ["OPENAI_API_KEY"]

instruction = "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user."

df = pd.read_csv('essay_prompts.csv', sep=';')

for index, row in df.iterrows():
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
        "role": "system",
        "content": instruction
        },
        {
        "role": "user",
        "content": row.essay_prompt_AAE
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
    df.at[index,'essay_from_AAE_prompt'] = response['choices'][0]['message']['content']
    df.to_csv('essay_prompts.csv', sep=';')


