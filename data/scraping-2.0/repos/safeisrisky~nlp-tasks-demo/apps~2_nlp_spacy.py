import streamlit as st
import pandas as pd
import openai
import json
import utils
from utils import get_message
from utils import get_system_prompt, get_companies_prompt, get_non_companies_prompt
import spacy_streamlit as ss
import spacy

if not utils.check_password():
    st.stop()

nlp_tasks_menu = [
    "Tokenization",
    "Word2Vec",
    "Named Entity Recognition",
    "Dependency Parser and POS",
    "Similarity",
]

nlp_models_menu = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]


@st.cache_resource
def get_nlp_model(model_type):
    "Loading NLP Model"
    nlp = spacy.load(model_type)
    return nlp


def main():
    if "nlp_model" not in st.session_state:
        st.session_state.nlp_model = "en_core_web_sm"
    if "rawtext" not in st.session_state:
        st.session_state.rawtext = None
    model_choice = st.sidebar.selectbox("NLP Model", nlp_models_menu)
    st.session_state.nlp_model = model_choice
    nlp = get_nlp_model(st.session_state.nlp_model)
    choice = st.sidebar.selectbox("Menu", nlp_tasks_menu)
    if choice == "Tokenization":
        st.subheader("Tokenization")
        input_tok = st.text_area("Enter Text", value=st.session_state.rawtext)
        st.session_state.rawtext = input_tok
        tok_button = st.button("Tokenize")
        if st.session_state.rawtext:
            docx = nlp(st.session_state.rawtext)
            ss.visualize_tokens(docx)
    elif choice == "Named Entity Recognition":
        st.subheader("Named Entity Recognition")
        input_ner = st.text_area("Enter Text", value=st.session_state.rawtext)
        st.session_state.rawtext = input_ner
        ner_button = st.button("NER")
        if st.session_state.rawtext:
            docx = nlp(st.session_state.rawtext)
            ss.visualize_ner(docx)
    elif choice == "Dependency Parser and POS":
        st.subheader("Dependency Parser and POS Tagging")
        input_dep = st.text_area("Enter Text", value=st.session_state.rawtext)
        st.session_state.rawtext = input_dep
        dep_button = st.button("Go")
        if st.session_state.rawtext:
            docx = nlp(st.session_state.rawtext)
            ss.visualize_parser(docx)
    elif choice == "Similarity":
        text1 = "The Company went bankrupt"
        text2 = "The company was involved in a financial scandal"
        ss.visualize_similarity(nlp, (text1, text2))
    elif choice == "Word2Vec":
        word_vec_input = st.text_input("Enter a word or phrase")
        tokens = nlp(word_vec_input)
        for token in tokens:
            st.write("Token:", token.text, "Vector Shape:", token.vector.shape)
            st.write(pd.DataFrame(token.vector))


main()
