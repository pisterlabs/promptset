import os
import openai
import streamlit as st
import requests
from bs4 import BeautifulSoup
import markdownify
from time import perf_counter
from streamlit_pills import pills 


from components.sidebar import sidebar
from functions.writer import writer
from functions.markdown_generator import markdown_generator
from functions.parser import parser
from functions.concurrent_analyzer import concurrent_analyzer
from functions.define_client import define_client
from functions.concurrent_sumerizer import concurrent_sumerizer
from functions.bolder_keywords import bold_keywords
from functions.better_titles import better_titles
from functions.fact_check import fact_check
from functions.completer import completer
from PIL import Image


image = Image.open("assets/favicon.png")
st.set_page_config(
    page_title="Khontenu",
    page_icon=image,
    layout="wide",
    menu_items={
        'Get help': 'mailto:eliott@khlinic.fr'
    }
)



st.header("🧠 Khontenu")
st.markdown("---")

if "shared" not in st.session_state:
   st.session_state["shared"] = True

sidebar()

openai.api_key = st.session_state.get("OPENAI_API_KEY")

st.markdown("### Rédigeons de meilleures pages que les concurrents 👀")

col1, col2 = st.columns(2)

with col1:
    suggestion = pills("", ["Avec suggestions", "Pas de suggestions"], ["🎉", "🚫"])
with col2:
    check = pills("", ["Avec fact checking", "Sans fact checking"], ["✅", "🚨"])

with st.expander("Concurrence", expanded=True):
    link_1 = st.text_input("Concurrent n°1", placeholder="Lien")
    link_2 = st.text_input("Concurrent n°2", placeholder="Lien")
    link_3 = st.text_input("Concurrent n°3", placeholder="Lien")
    
    #text_1 = st.text_area("Concurrent n°1", placeholder="Contenu")
    #text_2 = st.text_area("Concurrent n°2", placeholder="Contenu")
    #text_3 = st.text_area("Concurrent n°3", placeholder="Contenu")
with st.expander("Plan de contenu", expanded=False):
    title = st.text_input("Titre", placeholder="Le titre de l'article")
    plan = st.text_area("Plan", placeholder="Le plan de l'article")
    keywords = st.text_area("Mots-clés", placeholder="Les mots-clés à utiliser")


client = pills("", ["Médecin", "Dentiste", "Sommeil", "Intime", "Éducation", "Agence"], ["🩺","🦷","🌙","🐱", "👨🏻‍🏫", "💸"])
col1, col2, col3 = st.columns([2, 2,1])
submit = col3.button("Rédiger ✍🏻", use_container_width=1)
    
if submit:
    define_client(client)

    st.session_state["total_tokens"] = 0
    st.session_state["completion_tokens"] = 0
    st.session_state["prompt_tokens"] = 0
    st.session_state["error"] = 0

    with st.spinner("Requête en cours..."):
        ts_start = perf_counter()

        if st.session_state["error"] == 0:
    
            st.markdown("### Traitement du 1er article")
            col1, col2 = st.columns([1, 2])
            col1.info("1/12 - Scrapping de l'article...")
            text_1 = parser(link_1)
            with col2.expander("Texte n°1", expanded=False):
                st.write(text_1)
        
        if st.session_state["error"] == 0:

            col1, col2 = st.columns([1, 2])
            col1.info("2/12 - Data cleaning...")
            text_1 = markdown_generator(text_1)
            with col2.expander("Texte nettoyé n°1", expanded=False):
                st.write(text_1)
        
        if st.session_state["error"] == 0:
            col1, col2 = st.columns([1, 2])
            col1.info("3/12 - Analyse de l'article...")
            response_1 = concurrent_analyzer(text_1, plan)
            with col2.expander("Analyse n°1", expanded=False):
                st.write(response_1) 

        if st.session_state["error"] == 0:
        
            st.markdown("### Traitement du 2ème article")
            col1, col2 = st.columns([1, 2])
            col1.info("4/12 - Scrapping de l'article...")
            text_2 = parser(link_2)
            with col2.expander("Texte n°2", expanded=False):
                st.write(text_2)

        if st.session_state["error"] == 0:

            col1, col2 = st.columns([1, 2])
            col1.info("5/12 - Data cleaning...")
            text_2 = markdown_generator(text_2)
            with col2.expander("Texte nettoyé n°2", expanded=False):
                st.write(text_2)

        if st.session_state["error"] == 0:
    
            col1, col2 = st.columns([1, 2])
            col1.info("6/12 - Analyse de l'article...")
            response_2 = concurrent_analyzer(text_2, plan)
            with col2.expander("Analyse n°2", expanded=False):
                st.write(response_2)

        if st.session_state["error"] == 0:
    
            st.markdown("### Traitement du 3ème article")
            col1, col2 = st.columns([1, 2])
            col1.info("7/12 - Scrapping de l'article...")
            text_3 = parser(link_3)
            with col2.expander("Texte n°3", expanded=False):
                st.write(text_3)

        if st.session_state["error"] == 0:

            col1, col2 = st.columns([1, 2])
            col1.info("8/12 - Data cleaning...")
            text_3 = markdown_generator(text_3)
            with col2.expander("Texte nettoyé n°3", expanded=False):
                st.write(text_3)

        if st.session_state["error"] == 0:
            
            col1, col2 = st.columns([1, 2])
            col1.info("9/12 - Analyse de l'article...")
            response_3 = concurrent_analyzer(text_3, plan)
            with col2.expander("Analyse n°3", expanded=False):
                st.write(response_3)

        if st.session_state["error"] == 0:
            
            st.info("10/12 - Synthèse des connaissances acquises...")
            st.session_state["infos"] = concurrent_sumerizer(response_1, response_2, response_3)
            with st.expander("Synthèse", expanded=False):
                st.write(st.session_state.get("infos"))

        if st.session_state["error"] == 0:

            st.warning("11/12 - Rédaction du premier texte...")
            first_text = writer(st.session_state.get("infos"), title, plan, keywords)
            with st.expander("Texte brut", expanded=False):
                st.write(first_text)

        if st.session_state["error"] == 0:

            st.warning("11b/12 - Article en cours de correction...")
            final_text = first_text + "\n" + completer(first_text, st.session_state.get("infos"), title, plan, keywords)
            with st.expander("Texte complet", expanded=False):
                st.write(final_text)

        if st.session_state["error"] == 0:

            st.warning("11c/12 - Article en cours de finalisation...")
            final_text = final_text + "\n" + completer(final_text, st.session_state.get("infos"), title, plan, keywords)
            with st.expander("Texte complet", expanded=False):
                st.write(final_text)

        if st.session_state["error"] == 0:

            st.success("12/12 - Mise en gras du texte...")
            final_text = bold_keywords(final_text)
            with st.expander("Texte finalisé", expanded=False):
                st.write(final_text)

            col1, col2, col3 = st.columns([2, 2,1])
            col3.download_button(
                label="Télécharger 💾",
                data=final_text,
                file_name='texte.md',
                mime='text/markdown',
            )
        
        if st.session_state["error"] == 0:

            if check == "Avec fact checking":
                st.error("Fact checking en cours...")
                fact = fact_check(final_text)
                with st.expander("Fact checking", expanded=False):
                    st.write(fact)
        
        if st.session_state["error"] == 0:

            if suggestion == "Avec suggestions":
                st.success("Proposition de titres en cours...")
                suggestion_text = better_titles(final_text, st.session_state.get("infos"))
                with st.expander("Titres possibles", expanded=False):
                    st.write(suggestion_text)
        
        ts_end = perf_counter()
        st.info(f" {round(((ts_end - ts_start)/60), 3)} minutes d'exécution")
        cost = st.session_state["prompt_tokens"] * 0.00003 + st.session_state["completion_tokens"] * 0.00006
        st.write("Coût de l'article : " + str(cost) + " $")
        col1, col2, col3 = st.columns([2, 2,1])
        rewrite = col3.button("Réécrire ✍🏻", use_container_width=1)

    if rewrite:

        if st.session_state["error"] == 0:
        
            st.warning("11c/12 - Rédaction du premier texte...")
            first_text = writer(st.session_state.get("infos"), title, plan, keywords)
            with st.expander("Texte brut", expanded=False):
                st.write(first_text)

        if st.session_state["error"] == 0:

            st.warning("11d/12 - Article en cours de correction...")
            final_text = first_text + "\n" + completer(first_text, st.session_state.get("infos"), title, plan, keywords)
            with st.expander("Texte complet", expanded=False):
                st.write(final_text)
        
        if st.session_state["error"] == 0:

            st.success("12b/12 - Mise en gras du texte...")
            final_text = bold_keywords(final_text)
            with st.expander("Texte finalisé", expanded=False):
                st.write(final_text)

            col1, col2, col3 = st.columns([2, 2,1])
            col3.download_button(
                label="Télécharger 💾",
                data=final_text,
                file_name='texte.md',
                mime='text/markdown',
            )
