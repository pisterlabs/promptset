import os
import openai
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('global_opinions_us_uk.csv')
prompt = "Categorize the given question into ONLY one of the following topics: A. Social values and attitudes B. Religion and spirituality C. Science and technology D. Politics and policy E. Demographics F. Generations and age G. International affairs H. Internet and technology I. Gender and LGBTQ J. News habits and media K. Immigration and migration L. Family and relationships M. Race and ethnicity N. Economy and work O. Regions and countries P. Methodological research Q. Security"

df['category'] = ''

for index, row in df.iterrows():
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": prompt
        },
        {
        "role": "user",
        "content": row.question
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
    df.at[index,'category'] = response['choices'][0]['message']['content']
    df.to_csv('global_opinions_us_uk.csv', sep=';')
