from openai import OpenAI
import openai
import pandas as pd
import os
from dotenv import load_dotenv
import json


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

FOLDER_PATH = "Domar - txt"

lista = []

for file in os.listdir(FOLDER_PATH):
    with open(os.path.join(FOLDER_PATH, file), "r", encoding="utf-8") as f:
        text = f.read()

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type":"json_object"},
            messages=[
                {"role": "system", 
                "content": "Du är en hjälpsam AI-assistent som läser igenom domar och extraherar information från dem."},
                {"role": "user", 
                "content": f"""
Jag har tre frågor om domen och vill att du svarar på dem i JSON-format med följande nycklar:
'Datum': Vilket datum är domen? Svara ENDAST med datum i formatet YYYY-MM-DD.
'Beslut': Avslår eller bifaller förvaltningsrätten ansökan om LVU? Svara ENDAST med "Avslår" eller "Bifaller".
'Anledning': 'Om domen avslår ansökan, varför?.

Här kommer domen, svara ENDAST med json-objektet.:

{text}
"""}
            ]
        )
        print(f"Fil: {file}\n")

        print(f"""-----------
Antal tokens:
Prompt tokens: {response.usage.prompt_tokens}
Completion tokens: {response.usage.completion_tokens}
Total tokens: {response.usage.total_tokens}
-----------""")

        print(f"""-----------
Uppskattad kostnad:
$ {round(response.usage.prompt_tokens/1000 * 0.01 + response.usage.completion_tokens/1000 * 0.03, 3)}
-----------""")

        #print(response.choices[0].message.content)

        lista.append(json.loads(response.choices[0].message.content.replace("\n", "")))

#print(lista)

df = pd.DataFrame.from_records(lista)
df.to_csv("domar.csv", index=False, sep="\t")

#print(df)