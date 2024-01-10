import pandas as pd
import openai
#! replace api_list with real keys
api_list = ['org-','sk-']


df = pd.read_csv('nasdaq_screener_1697135861885.csv')

industry = set(df['Industry'].to_list())
sector = set(df['Sector'].to_list())

def answer_questions(task, input, system):
    openai.organization = api_list[0]
    openai.api_key =  api_list[1]
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", temperature = 0.5,
    messages=[{'role': 'system', 'content': system},
     {'role': 'user', 'content': f'task = ({task}), input = ({input})'}]
    )
    response_var = completion["choices"][0]["message"]["content"]
    return response_var


task = "Your task is to classify Sector and Industry, you can find current categories to classify, your output should contain " \
       "two dictionaries, first with Sector, second with Industry. I will provide lists of Sectors And Industries, You should reduce and minimaze the " \
       "number of categories"

print(answer_questions(task = task, input = f"sectors = {sector}, industries = {industry}", system = task))