import requests
import openai
import bs4
import requests
import streamlit as st

class TextProcessor:

    def __init__(self, api_key = st.secrets["api-key"]):
        self.api_key = api_key
        openai.api_key = self.api_key

    def translate(self, trad):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": "Tu dois traduire ce texte en anglais"
            },
            {
            "role": "user",
            "content": trad
            }
        ],
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        return response['choices'][0]['message']["content"]
    
    def summary(self, text):
        reponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": "Tu dois résumer ce texte en utilisant au maximum 50 mots"
            },
            {
            "role": "user",
            "content": text
            },
        ],
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        return reponse['choices'][0]['message']["content"] 


    def imagine(self, text):
        response = openai.Image.create(
            prompt= text,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        return image_url
    

    def code(self, code):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": "Analyse ce code et corrige moi les erreurs"
            },
            {
            "role": "user",
            "content": code
            }
        ],
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        return response['choices'][0]['message']["content"]

    def actu(self, query):

        scrap = requests.get(f'https://www.bing.com/news/search?q={query}').text
        soup = bs4.BeautifulSoup(scrap, 'html.parser')

        actu = ' '.join(["- Actualité : " + link.text+ ' \n'  for link in soup.find_all('a', 'title')])

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                "content": f"Tu es un rédacteur web qui synthétise l'actualité en 100 mots sur la thématique '{query}' Tu fais des liaisons entre les articles avec des mots tel que 'mais', 'donc', 'or', 'par contre', 'en revanche', 'en effet', 'cependant', 'toutefois', 'par ailleurs', 'par contre', 'par contre, 'enfin'"},
                {"role": "user",
                "content": "Voici la liste des actualités à synthétiser :" + actu},
            ],
            max_tokens=100,
            temperature=0.3,
        )

        return response['choices'][0]['message']["content"]

    
    def json_format(self, url):

        response = requests.get(url).text
        soup = bs4.BeautifulSoup(response, "html.parser")
        text = soup.text.replace("\n", " ").replace("\t", " ").replace(' ', '')

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "system",
                "content": "J'ai besoin que tu me convertisse ce texte au format json"
                },
                {
                "role": "user",
                "content": text
                }
            ],
            temperature=0.3,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['message']["content"]

    
    
    
    
