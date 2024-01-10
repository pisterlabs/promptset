from classes.arc_classique import ARC_Classique

import streamlit as st 
import os
import openai
import pandas as pd

st.set_page_config(page_title="Adcom - ARC", page_icon="favicon.ico", layout="wide", initial_sidebar_state="expanded")

# DEV
# openai.api_key = os.getenv("OPENAI_API_KEY")

# PROD
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title('Assistance à la Rédaction de Contenu - Classique')

arc_classique = ARC_Classique()
df_examples = pd.read_json(r'prompt_examples.json')

with st.sidebar:
    st.write('Version de chatGPT utilisée : gpt-4-1106-preview')
    st.write("Quelques documentations : %s" % ", ".join(["https://beta.openai.com/examples", "https://www.webrankinfo.com/dossiers/conseils/chatgpt-seo", "https://learnprompting.org/docs/intro", "https://flowgpt.com/"]))

upper_col1, upper_col2 = st.columns(2)
with upper_col1:
    user_word_number_input = st.slider("Choisissez la longueur à générer :", 50, 2000, 750, 50, help="Un token correspond plus ou moins à une syllabe. 'Chat' = 1 token, 'Montagne' = 3 tokens, 'Sarkozy' = 4 tokens car mot peu commun.")
with upper_col2:
    user_temperature_input = st.slider("Choisissez la température (originalité) :", 0.0, 1.0, 0.5, 0.05, help="Une température plus élevée signifie que le modèle prendra plus de risques. Essayez 0.9 pour des applications plus créatives et 0 pour celles avec une réponse bien définie. Avec une température de 0.9, il est probable que les résultats soient en anglais.")

lower_col1, lower_col2 = st.columns(2)
with lower_col1:
    # action_selector = st.multiselect("Typologies de prompt", arc_classique.df_examples['Type'].unique(), ['Mots-clés', 'Syntaxe'])
    action_selector = st.multiselect("Typologies de prompt", df_examples['Type'].unique(), ['Mots-clés', 'Syntaxe'])
    
with lower_col2:
    # input_selector = st.selectbox("Prompts pré-remplis en fonction des typologies sélectionnées", arc_classique.df_examples[arc_classique.df_examples['Type'].isin(action_selector)].sort_values(by='Utilisation'))
    input_selector = st.selectbox("Prompts pré-remplis en fonction des typologies sélectionnées", df_examples[df_examples['Type'].isin(action_selector)].sort_values(by='Utilisation'))

# user_text_input = st.text_area(label="", value=arc_classique.df_examples[arc_classique.df_examples['Utilisation'] == input_selector]['Saisie'].values[0] if input_selector else "")
user_text_input = st.text_area(label="", value=df_examples[df_examples['Utilisation'] == input_selector]['Saisie'].values[0] if input_selector else "")

if user_text_input:
    if st.button('Générer le texte'):
        st.write(arc_classique.generate_answer(user_text_input, user_word_number_input, user_temperature_input))
else:
    st.warning("Merci de saisir un prompt.")
    