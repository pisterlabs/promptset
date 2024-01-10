# pip install langchain
# pip install huggingface_hub
# pip install streamlit
# pip install python-dotenv

# Travail realisé par Aymen Fourati dans le cadre du case study proposé par l'entreprise Malou ✍️


################################################################################################
# Dependances :
# Langchain est un framework qui permet d'integrer tres rapidement plusieurs types de LLMs
# Dans notre cas, nous avons utilisé la classe HuggingFaceHub qui permet d'interagir avec les modèles de HuggingFace
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


import json
import re
import os
from dotenv import load_dotenv

# Nous utilisons streamlit pour créer une interface utilisateur
import streamlit as st

load_dotenv()

# Cette fonction permet d'extraire la réponse du modèle puisque la forme de la reponse est generalement JSON
def extract_reply(input_string):

    match = re.search(r'"reply":\s*"([^"]+)"', input_string)

    if match:
        return match.group(1)
    match = re.search(r'"reply":\s*"(.*?)"(?=[^"]*$)', input_string)
    if match:
        return match.group(1)
    else:
        return input_string
    
# Il faut mettre votre clé HugginFace dans le fichier .env 
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

template = """{{"keywords": {keywords},"{comment}","comment": ""}}"""

prompt = PromptTemplate(template=template, input_variables=["comment","keywords"])

# Le repo_id est le nom du repo sur HuggingFace foufou26 c'est moi haha ! 
# Vous pouvez consulter mon profile avec mon nom apartir de ce URL : https://huggingface.co/foufou26
repo_id = "foufou26/malin"

llm = HuggingFaceHub(
    repo_id=repo_id,model_kwargs={"temperature": 0.5, "max_length": 500}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)


st.title("Malin 🤖 - v0.1")

user_comment = st.text_input("Plug in your comment here 🤌")
user_keywords = st.session_state.get("array_keywords", ["restaurant"])

# Text input for new keyword
new_keyword = st.text_input("Add a new keyword:")
    
# Add new keyword to the list when the "Add" button is clicked
if st.button("Add") and new_keyword:
        user_keywords.append(new_keyword)
        st.session_state.array_keywords = user_keywords


if st.button("Generate"):
    st.write(user_keywords)
    if user_comment=="" and user_keywords==[]:
        st.write("Please enter a comment and keywords")
    else:
        st.write(extract_reply(llm_chain.run(comment=user_comment, keywords=user_keywords)))


st.markdown("""*Travail realisé par Aymen Fourati dans le cadre du case study proposé par l'entreprise Malou ✍️*""")