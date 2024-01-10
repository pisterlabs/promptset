from openai import OpenAI
from Sentiment import Sentiment
import os
from dotenv import load_dotenv



load_dotenv()

def chat_request_tag(comment:str):
    openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Proszę Wypisać które tagi pasują do podanego komentarza: JEDZIENIE - WYPOSAŻENIE - OBSŁUGA - ATMOSFERA - MENU - CENA. tagi wypisz po przecinku bez spacji"},
            {"role": "user", "content": comment},
        ],
        temperature=1,
    )
    tag_tab = []
    tag = response.choices[0].message.content
    if(tag.find("JEDZIENIE") != -1):
        tag_tab.append("JEDZENIE")
    if(tag.find("WYPOSAŻENIE") != -1):
        tag_tab.append("WYPOSAŻENIE")
    if(tag.find("OBSŁUGA") != -1):
        tag_tab.append("OBSŁUGA")
    if(tag.find("ATMOSFERA") != -1):
        tag_tab.append("ATMOSFERA")
    if(tag.find("MENU") != -1):
        tag_tab.append("MENU")
    if(tag.find("CENA") != -1):
        tag_tab.append("CENA")
    return tag_tab
        

        
def chat_request_sentiment(comment:str):
    openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Proszę zanalizować czy komentarz dotyczący restauracji jest pozytywny, negatywny, neutralny, czy jest nieistotny. Lista odpowiedzi [GOOD, BAD, NEUTRAL, IRRELEVANT]"},
            {"role": "user", "content": comment},
        ],
        temperature=0,
    )
    value = response.choices[0].message.content
    return value

    
def chat_request_analysis(comment:str):
    openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "do zanalizowanych danych dodaj komentarz dla każdego miesiąca będący podsumowaniem zmian od ostatniego miesiąca , np.  pogorszył się stan jedzenia , na koniec dodaj opis tendencji w formie jednego zdania dla każdego z tagów"},
            {"role": "user", "content": comment},
        ],
        temperature=1,
    )
    value = response.choices[0].message.content
    return value


# data =chat_request_tag("jedzenie było pyszne, ale bardzo i to bardzo drogie")
# data =chat_request_sentiment("jedzenie było pyszne, ale bardzo i to bardzo drogie")
# print(data)