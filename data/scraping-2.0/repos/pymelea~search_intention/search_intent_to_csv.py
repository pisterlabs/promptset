import json
from os import path

import openai
import pandas as pd

import config


def get_categorization(query):

    #  Get OpenAI
    openai.api_key = config.API_KEY

    # consulta = ""
    conversacional = 'Sorry to hear that. Maybe try doing something that makes you happy, like hanging out with friends or doing a fun activity.'
    generacional = 'How can I create a marketing campaign that reflects my brand values?'
    
    # prompt = f'Tu eres un asistente de chatbot y dadas las siguientes sentencias: \n \n {conversacional} \n \n y \n \n {generacional} \n \n : \
    #         dime de esta nueva sentencia \n \n {query} \n \n  a que tipo de sentencia corresponde.'
    prompt = f'Tu eres un asistente de chatbot y dadas las siguientes sentencias: \n \n {conversacional} \n \n y \n \n {generacional} \n \n : \
            dime de esta nueva sentencia \n \n {query} \n \n  a que categoría de sentencia corresponde de (Conversation, Consulta , Generación o Campaigns) en una sola palabra, en ingles'

    categorization = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        max_tokens=1500,
        temperature=0.7,
    )
    return categorization


def categorize_keywords(categorization):
    category = json.dumps(categorization).encode('utf8').decode()
    category_json = json.loads(category)
    resp = category_json[0]['choices'][0]['text']
    keywords_categorizadas = resp.split('\n')
    return keywords_categorizadas[-1:]


# Create a Pandas DataFrame with the keywords
def create_dataframe(query, intention):
    tabla = pd.DataFrame({'query': query, 'Intention': intention})
    return tabla


def save_dataframe_csv(tabla):
    tabla.to_csv('docs/results.csv')


if __name__ == '__main__':
    query = input('Enter your query: ')
    # query = 'We need to increase sales for our products.'
    categorization = (get_categorization(query),)
    intention = categorize_keywords(categorization)
    print(intention)
    save_dataframe_csv(create_dataframe(query, intention))

