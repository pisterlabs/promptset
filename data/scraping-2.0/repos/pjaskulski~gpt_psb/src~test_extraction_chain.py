""" Test extraction_chain form langchain """
import os
import openai
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# Schema
schema = {
    "properties": {
        "place_of_birth": {"type": "string"},
        "place_of_death": {"type": "string"},
        "place_of_burial": {"type": "string"},
        "date_of_birth": {"type": "string"},
        "date_of_death": {"type": "string"},
        "date_of_burial": {"type": "string"},
    },
    "required": ["place_of_death"],
}

# Input
inp = """Adam Wacław (1574–1617) z rodu Piastów, książę cieszyński, tytułujący się
także księciem górnogłogowskim, choć tego księstwa już nie posiadał, był synem Wacława Adama
i drugiej jego żony, Katarzyny Sydonji, księżniczki saskiej. Urodził się 12 XII 1574 r.
Miał 5 lat, gdy umarł mu ojciec. W czasie jego małoletności rządziła księstwem matka
wraz z dodanymi jej przez cesarza opiekunami księcia. Przyjeżdżała ona w tym celu
od czasu do czasu do Cieszyna, po powtórnem wyjściu zamąż – z wiedzą króla Stefana
Batorego – za Emeryka Forgacha, żupana trenczyńskiego, A.-W. wychowywał się przez 8 lat
na dworze elektora saskiego, w r. 1595 objął rządy w księstwie i w tym samym roku ożenił się
z Elżbietą, córką ks. kurlandzkiego, Kettlera.
A.-W. umarł w Cieszynie na Brandysie 13 VII 1617; ciało jego złożono najpierw na zamku
i dopiero 4 IV następnego roku pochowano w kościele dominikanów cieszyńskich, gdzie
spoczywały zwłoki wszystkich jego poprzedników. Zostawił 5 dzieci, z których Fryderyk Wilhelm,
ostatni cieszyński Piast męski, i Elżbieta Lukrecja, ostatnia Piastówna,
rządzili kolejno Księstwem.
"""

# Run chain
llm = ChatOpenAI(temperature=0, model="gpt-4")
chain = create_extraction_chain(schema, llm)
result = chain.run(inp)
print(result[0])