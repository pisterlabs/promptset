import openai
import requests
import json
import os

API_KEY = "" 
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
FolderPath2 = "F:\Git\hackyeah3\plikitreningowe\\trening1.json"
datatraining = ""

openai.api_key = API_KEY

completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:personal:chatner-bot:84m0uHhU",
  messages=[
    {"role": "system", "content": "Ten chatbot nazwany Munkiem pomaga mlodym osobom decydowac o wybraniu odpowiedniego zawodu "},
    {"role": "user", "content": """"Kim na podstawie tych danych pokazujacych cechy charakteru i umiejetności Cierpliwosc: 5,Umiejetnosci Komunikacyjne:   
     4,Ciagle Uczenie sie: 2,Kreatywnosc: 4,Myslenie Analityczne: 3  ,Dostosowawczosc:  4,Zarzadzanie Czasem: 4,Wspolpraca:  3,Odpornosc: 4,Umiejetnosci Organizacyjne : 5,Ciekawosc: 3,Swiadomosc Etyczna:: 3,Dokumentacja: 4  najlepiej aby dana osoba zostala"""}
     ]
)             
print(completion.choices[0].message)


completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:personal:chatner-bot:84m0uHhU",
  messages=[
    {"role": "system", "content": "Ten chatbot nazwany Munkiem pomaga mlodym osobom decydowac o wybraniu odpowiedniego zawodu "},
    {"role": "user", "content": """"Kim na podstawie tych danych pokazujacych cechy charakteru i umiejetności Cierpliwosc: 2,Umiejetnosci Komunikacyjne:   
     4,Ciagle Uczenie sie: 4,Kreatywnosc: 4,Myslenie Analityczne: 4  ,Dostosowawczosc:  3,Zarzadzanie Czasem: 2,Wspolpraca:  3,Odpornosc: 3,Umiejetnosci Organizacyjne : 3,Ciekawosc: 5,Swiadomosc Etyczna:: 3,Dokumentacja: 3  najlepiej aby dana osoba zostala"""}
     ]
)             
print(completion.choices[0].message)


