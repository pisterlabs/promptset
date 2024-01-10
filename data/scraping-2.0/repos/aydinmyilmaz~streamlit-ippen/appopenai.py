"""
Todo list app
"""

import streamlit as st
import openai
import os
import json


def app():

    def recipeGenerator(name_of_dish, ingredients):
        openai.organization = 'ippen'
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = openai_api_key
        headline = "Schreiben Sie ein Rezept basierend auf diesen Zutaten und Anweisungen:\n\n"
        name_of_dish = name_of_dish 
        ingredients = ingredients + "\Anweisungen:\n"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt= headline + name_of_dish + ingredients,
            temperature=0.3,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)

        return response

    def summary(text):
        openai.organization = 'ippen'
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = openai_api_key
        text = text + "\n\nTl;dr"
        response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=text,
        temperature=0.7,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
        return response['choices'][0]['text']


    add_selectbox = st.sidebar.selectbox(
    "Select NLP Function?",
    ("Recipe Generator", "Summarization", "QA")
)
    if add_selectbox == "Recipe Generator":
        st.header("Recipe Generator")
        option = st.selectbox(
                    'Select an option',
                    ( '', 
                     'Show Sample Recipe', 
                     'Generate Own Recipe'))
        
        if option == 'Show Sample Recipe':
            ingredients = """
                    1 Zwiebel, gewürfelt
                    2 Knoblauchzehen, fein gehackt
                    2 TL gemahlener Koriander
                    1 TL Kreuzkümmel (Cumin)
                    1 TL Curcuma
                    1 Prise Chiliflocken
                    1 Prise Salz
                    3 cm Ingwer, fein gehackt
                    4 Tomaten, gewürfelt
                    4 EL rote Linsen
                    4 EL Kokosnusscreme
                    250 g Brokkoli, in Röschen
                    1 Dose Kichererbsen, abgetropft
                    100 g frischer Babyspinat
                    Saft von 1/2 Zitrone
                    Zum Servieren:
                    3 EL Naturjoghurt
                    2 EL frischer Koriander, gehackt""" 
            name_of_dish = "Kichererbsen-Curry"
            recipe = """ 1. Zwiebel und Knoblauch in einer Pfanne anbraten.
                        2. Koriander, Kreuzkümmel, Curcuma, Chiliflocken und Salz hinzufügen und kurz mitbraten.
                        3. Ingwer, Tomaten, Linsen, Kokosnusscreme und Brokkoli hinzufügen.
                        4. Alles ca. 15 Minuten köcheln lassen.
                        5. Kichererbsen und Babyspinat hinzufügen und noch 5 Minuten köcheln lassen.
                        6. Mit Zitronensaft abschmecken.
                        7. Mit Joghurt und Koriander garnieren und servieren."""
            st.write('Name of the dish is', name_of_dish)
            st.write('**ingredients**\n\n', ingredients)

        if option == 'Generate Own Recipe':
            st.title("OpenAI Recipe Generator")
            name_of_dish = st.text_input("Enter name of the dish", "", key="1")
            st.write('Name of the dish is', name_of_dish)
            ingredients = st.text_area("Enter ingredients","", key="2")
            st.write('**ingredients**\n\n', ingredients)
        
        if st.button('Show Recipe'):
            recipe = recipeGenerator(name_of_dish, ingredients)
            st.write('**eat at your own risk!!!**\n\n', recipe['choices'][0]['text'])
        
    elif add_selectbox == "Summarization":
        st.header("Summary Generator")
        
        # path = "data_1000.json"
        # with open(path, 'r') as json_file:
        #     data = json.load(json_file)
        # st.sidebar.title('Selection Menu')
        # st.sidebar.header('Select Parameters')
        # idx = str(st.sidebar.number_input('Insert an index number between 0-1000 to select a random Story', min_value=0, max_value=10000, value=0))
        # st.write('**Online Id:**', data[idx]['meta']['online_id'])
        # text = data[idx]['meta']['text'][0:500]
        
        text = st.text_area("Enter a text in german to summarize","", key="1")
        st.write('**Text:**', text[0:500])
        
        if st.button('Show Summary'):
            res_summary = summary(text[0:500])
            st.write('**Summary:**', res_summary)

    else:
        st.write('coming soon...')
