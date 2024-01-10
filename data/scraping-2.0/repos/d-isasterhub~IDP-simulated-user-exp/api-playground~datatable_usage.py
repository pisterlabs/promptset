import openai
import os
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]


TEMPLATE = [
    {"role": "system", "content": "You are a person who knows nothing. Act accordingly."}, 
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": "What is 3+3?"}
]

data = [{'name': "x", 'insert': "I don't know."}, 
        {'name': "y", 'insert': "5"}] 

df = pd.DataFrame(data) 

def get_answer(row):
    query = TEMPLATE
    query[2]['content'] = row['insert']
    response = openai.ChatCompletion.create(
        model = "gpt-4-vision-preview",
        max_tokens = 300,
        messages = query
    )
    return response["choices"][0]["message"]["content"]
    
df['answer'] = df.apply(get_answer, axis=1)

print(df)