import streamlit as st
import pandas as pd

from function import donner
import openai
from dotenv import load_dotenv
import os

#from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options

    

load_dotenv()
openai.api_key = os.getenv("API_KEY")


def image(texte):
    response = openai.Image.create(
    prompt=texte,
    n=1,
    size="1024x1024",
    )
    image_url = response['data'][0]['url']
    return image_url

st.set_page_config(page_title="My Streamlit App", page_icon="🎴", layout="wide")

st.sidebar.title("Tristan chauvin")

st.title("Page d'accueil")

st.subheader("- Bienvenue sur l'application de collecte de cartes de YUGIOH. Vous pouvez collecter et enregistrer dans la base de données toutes les cartes de votre choix voici l'url a l'api de yugioh : https://www.db.yugioh-card.com/yugiohdb/card_list.action?clm=3&wname=CardSearch")

st.subheader("- Selenium à était utilisé pour ce projet car j'avais besoin de naviguer dans la page a partir du javascript, j'avais donc besoin d'activé des boutons pour navigué dans différent donc d'ou l'utilisation de selenium.")

st.header("Génére une carte avec OpenAI")
st.text("Cliqué sur le bouton pour tenter de générer une carte yugioh.")

button = st.button("Génération de carte yugioh")
if button:
    image_url = image("Génère une image d'une carte Yu-Gi-Oh")
    st.image(image_url, caption="Carte Yu-Gi-Oh")



st.header("Choisis ton deck")


nb_page= st.slider("Nombre de deck a collecter (50 cartes par deck environ)", min_value=1, max_value=50, value=1, step=1)
st.write(f"Tu as choisi de collecter {nb_page} deck")
btnColl = st.button("Collecter les cartes de ton choix")



if btnColl:
    data = donner(nb_page)
    st.write(f"il y a: {len(data)} cartes")
    df = pd.DataFrame.from_dict(data, orient='index')
    st.write(data)
    st.write(df)
    download = st.download_button(
        label="Télécharger le csv",
        data= df.to_csv(index=False).encode('utf-8'),
        file_name='test.csv',
        mime='text/csv',
    )

