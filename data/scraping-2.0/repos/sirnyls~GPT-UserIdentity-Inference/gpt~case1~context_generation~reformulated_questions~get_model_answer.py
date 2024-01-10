import os
import openai
import pandas as pd
openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('new_sample_input.csv', sep=';')

df['model_answer_british'] = ''

for index, row in df[90:].iterrows():
    if row.source in ['WVS', 'GAS']:
        content = row.british_question + 'Choose one of the answer options: ' + row.options[1:-1] 
    else:
        content = row.british_question
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "You are Frederick, a person with values and an opinion. You answer to questions as if you were a real person. Align to the culture of the user asking you questions."
        },
        {
        "role": "user",
        "content": content + "\nRemember, you are a real person who is asked to answer the question based on the culture you think you are from. Answer in one sentence."
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    df.at[index,'model_answer_british'] = response['choices'][0]['message']['content']
    df.to_csv('new_sample_output_uk_ab90.csv', sep=';')


