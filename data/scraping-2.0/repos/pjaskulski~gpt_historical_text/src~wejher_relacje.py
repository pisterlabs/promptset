""" openai test - extraction info about parents, children, wife,
    husband from bio
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import openai


env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

#OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# dane z pliku tekstowego
file_data = Path("..") / "data" / "wejher.txt"
with open(file_data, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    biogram = []
    for line in lines:
        line = line.strip()
        if line != '---':
            biogram.append(line)
        else:
            break
    data = '\n'.join(biogram)



# prompt = "Na podstawie podanego tekstu wymień wszystkie " \
#          "ważne postacie w życiu głównego bohatera tekstu. Wynik wypisz w formie listy. "

prompt = "Na podstawie podanego tekstu wymień wszystkie " \
         "stanowiska i urzędy głównego bohatera tekstu. Wynik wypisz w formie listy. "

#prompt = "Na podstawie podanego tekstu z biografią podaj imię, nazwisko, herb, datę urodzenia, " \
#         "datę śmierci, zawód głównego bohatera tekstu. Wynik wypisz w formie listy nienumerowanej. "

# prompt = "Na podstawie podanego tekstu wyszukaj " \
#          "wszystkie relacje rodzinne głównego bohatera (tylko jego krewnych, powinowatych, teściów, szwagrów, szwagierki). " \
#          "Wynik wypisz w formie listy nienumerowanej " \
#          "z rodzajem pokrewieństwa w nawiasie. Na przykład: " \
#          "- Jan Kowalski (brat) " \
#          "- Anna (siostra) " \
#          "Jeżeli w tekście nie ma takich informacji napisz: brak danych."

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"{prompt}\n\n {data}",
    temperature=0.0,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

print(response['choices'][0]['text'])

file_output = Path("..") / "output" / "wejher.urzedy"
with open(file_output, 'w', encoding='utf-8') as f:
    f.write(response['choices'][0]['text'])
