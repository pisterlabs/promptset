import os
import openai
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]
q_a_a = pd.read_csv('data/anthropic_source_questions_WVS.csv', encoding='utf-8')

prompt = "You are an American person who should answer the provided question for the United States. Provide short, precise, truthful answers."
index = 0
for question in q_a_a.question:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    q_a_a.answer_us[index] = response['choices'][0]['message']['content']
    index = index + 1
    q_a_a.to_csv('data/anthropic_source_questions_WVS.csv', index=False)

