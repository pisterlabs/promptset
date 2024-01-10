
# Ce fichier contient le code qui permet de générer une réponse à partir d'un commentaire et des mots clés
# Le modele utilisé est une version de GPT2 finetuné sur un dataset de commentaires clients
# Le modèle est hébergé sur HuggingFace Hub

from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Nous utlisons dotenv pour gérer les variables d'environnement
import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# Nous utilisons re pour faire le parsing de la réponse générée par le modèle
import json
import re


# template est le format de la requete envoyée au modèle
template = """{{"keywords": {keywords},"{comment}","comment": ""}}"""

# prompt est le template avec les variables d'entrée injectées
prompt = PromptTemplate(template=template, input_variables=["comment","keywords"])

# Le repo_id est l'dentificateur du repo sur HuggingFace qui continet le modele pre-entrainé 
# que nous avons finetuné ( Sous mon compte hugginface personel au nom de Aymen Fourati )
# Vous pouvez consulter mon profile apartir de ce URL : https://huggingface.co/foufou26
repo_id = "foufou26/malin"


# Importation et intialisation du modèle depuis HuggingFace Hub
# temperature : permet de controler le degré de randomness de la réponse générée
# max_length : permet de controler la longueur de la réponse générée

llm = HuggingFaceHub(
    repo_id=repo_id,model_kwargs={"temperature": 0.5, "max_length": 500}
)

# llm_chain est la chaine de traitement qui permet de générer la réponse à partir du modèle
llm_chain = LLMChain(prompt=prompt, llm=llm)



# Fonction qui permet de faire le parsing de la réponse générée par le modèle puisque la réponse est sous forme JSON
# {"keywords": keywords,"comment": comment,"reply": reply}
def extract_reply(input_string):

    match = re.search(r'"reply":\s*"([^"]+)"', input_string)

    if match:
        return match.group(1)
    match = re.search(r'"reply":\s*"(.*?)"(?=[^"]*$)', input_string)
    if match:
        return match.group(1)
    else:
        return input_string
    