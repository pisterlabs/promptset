import math
import requests
import spacy
import json
import os
import pandas as pd

from openai import OpenAI

client = OpenAI()

# spacy, para obtener entidades relacionadas al texto
nlp = spacy.load("es_core_news_sm")

test = pd.read_csv('test/dataset-test-desafio2.csv')
headers = {"Authorization": "Bearer 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"}

# Paso 1, traducir la noticia al ingles
API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-es-en"

def query_translate(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def translate_to_en(new: str) -> str:
    size_of_new = len(new)
    iter = math.ceil(size_of_new / 512)
    translated = ''
    for i in range(0, iter):
        start_index = i * 512
        end_index = (i + 1) * 512
        fragment = new[start_index:end_index]

        output = query_translate({
            "inputs": f'{fragment}',
        })
        
        translated += str(output[0]['translation_text'])
    return translated


# translated = translate_to_en(new)

prompt_0 = """Read the following news article and tell me what is the main event and where does is it take place.
In some cases the most important event occurs in an organization. If so, identify it and complement the answer.
With this additional information you can improve the response to location, consider the following information: 
LOC = Locations; ORG = Companies, agencies, institutions; PER = people. Sorted by priority. Try the be the more specific in the location.
Please respond with a JSON containing two fields: {'event': 'str', 'location': 'str'}, and in spanish language."""

prompt_1 = """Read the provided news article and extract information about the main event and its location. 
If the primary event takes place within an organization, identify it and provide additional details to enhance the location response. 
For improved location specificity, take into account the following information: LOC = Locations; ORG = Companies, agencies, institutions; PER = People. 
Prioritize information accordingly. Ensure your response is in JSON format with two fields: {'event': 'str', 'location': 'str'}, and in language spanish."""

prompt_2 = """
Read the provided news article and provide insights into the primary event and its location. If the key event takes place within an organization, please identify and elaborate on it. 
Additionally, enhance the response by incorporating the following location format: "number street, town, city, district, region, country." or "organization, city, district, region, country.".
In case any of these details are missing, simply omit them.

For better contextualization, utilize the following markers:
- LOC for Locations
- ORG for Companies, agencies, institutions
- PER for People

Prioritize information based on significance, and aim for specificity, especially when identifying the location.
Your response should be structured in JSON format, featuring two fields: {'event': 'str', 'location': 'str'}, and in spanish language.'"""

prompt_3 = """Read the following news article and provide information on the main event and its location. 
If the primary event occurs within an organization, please identify it and include relevant details in your response.

To assist with the location, consider the following information:
- LOC: Locations
- ORG: Companies, agencies, institutions
- PER: People

Please prioritize these elements in your response and be as specific as possible about the location. 
The location format should be: "number street, town, city, district, region, country." If any of these elements is missing, omit it. 
If there are multiple locations, return a JSON with an array of found locations.

Respond with a JSON containing two fields: {'event': 'str', 'location': 'str'}, and in spanish language.'
"""

prompt_4 = """
Read the following news article and provide information about the main event and its location. 
If the primary event involves an organization, identify and include details about it in your response.

To enhance the accuracy of the location information, consider the following format: 
- LOC = Locations (numerical address, street, population, city, municipality, region, country).
- ORG = Companies, agencies, institutions.
- PER = Individuals.

Sort the entities by priority and aim for specificity in the location details. 
If there are multiple locations mentioned, prioritize the one most relevant to the context.
e.g. Plaza de Mayo, Buenos Aires, Argentina; Paseo de la Reforma; Mexico City, Mexico.
Please respond with a JSON containing two fields: {'event': 'str', 'location': 'str'}, and in spanish language.
"""

prompt_5 = """
Read the following news article and provide information about the main event and its location. 
If the primary event involves an organization, identify and include details about it in your response.

To enhance the accuracy of the location information, consider the following format: 
- LOC = Locations related to rivers in Chile (numerical address, street, population, city, municipality, region, country).
- ORG = Companies, agencies, institutions.
- PER = Individuals.

Sort the entities by priority, and aim for specificity in the location details. 
If there are multiple locations mentioned, prioritize the one most relevant to the context, especially if it's related to a Los Ríos, Chile.

Please respond with a JSON containing two fields: {'event': 'str', 'location': 'str'}, and in spanish language.
"""

prompt_2_1 = """Read the following news article and identify the main event and its location.
If the main event occurs within an organization, please specify it and provide additional context.
To enhance the accuracy of location information, consider the following format for location:
'number street, population, city, commune, region, country.'
If any of these elements is missing, simply omit it. If there are multiple locations in the article, return the one most relevant to the context.
If no specific location is found, default to 'Valdivia, Los Rios, Chile.'
Additionally, utilize the following abbreviations: LOC for locations, ORG for companies, agencies, institutions, and PER for people, sorted by priority.
Strive for specificity in identifying the location.
Please respond with a JSON containing two fields: {'event': 'str', 'location': 'str'}, and in spanish language."""

def get_ents(new: str) -> str:
    doc = nlp(new)
    doc_ents = {"ORG": [], "LOC": [], "PER": []}

    for ent in doc.ents:
        if ent.label_ in ["ORG", "LOC", "PER"]:
            doc_ents[ent.label_].append(ent.text)

    doc_ents_str = "\n".join([f"{key}: {', '.join(value)}" for key, value in doc_ents.items()])
    return doc_ents_str

def call_gpt(new:str, prompt: str, ents: str):
    prompt = f"{prompt}\nNews article:\n{new}.\nAdditional information:\n{ents}"
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
                {"role": "system", "content": "You are an expert at identifying addresses in news articles."},
                {"role": "user", "content": prompt},
            ],
    )
    return response.choices[0].message.content

test = pd.read_csv('test/dataset-test-desafio2.csv')
# # new = test['text'][6]
# # ents = get_ents(new)
# # response = call_gpt(new, prompt_2, ents) # retorna un json {event: str, location: str}
# # print(response)

event_list = []
location_list = []
# Iterar a través de las filas de test
for index, row in test.iterrows():
    new = row['text']
    ents = get_ents(new)
    response = call_gpt(new, prompt_3, ents)
    main_event_location = json.loads(response)
    # Agregar los resultados a las listas
    event_list.append(main_event_location['event'])
    location_list.append(main_event_location['location'])
    if index % 10 == 0:
        print(index)
test['event'] = event_list
test['location'] = location_list

test.to_csv('test_3.csv')
