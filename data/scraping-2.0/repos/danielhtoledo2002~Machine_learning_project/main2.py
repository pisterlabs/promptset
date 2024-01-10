# Manage enviroment  variables
import os

# Tool
import pandas as pd

# web framework
import streamlit as st

# OPEN AI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from Models2 import tesne, logistic, svg, cosine_large, svg2
from DeepLearningModels import load_CNN15k



df3 = pd.read_csv("OpenAi/amlo_clasify_chatpgt_15k.csv")
select_clas = ""

with st.sidebar:
    st.write(" # Configuration")
    st.write(
        "We train three types of models, one that  was classified by human  other that chat-gpt-3.5 did with all data and the last one with only 15k with chat gpt."
    )
    clas = st.radio(
        "Select  which clasification you want to use",
        ["Chat gpt 15k:computer:"],
        index=None,
    )
    select_clas = clas
    if select_clas == "Chat gpt 15k:computer:":
        selected = st.multiselect(
            "Columns SVG 2",
            svg2.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        third_table3 = svg2.clasification_rep()[selected]

        selected = st.multiselect(
            "Columns CNN 2",
            load_CNN15k.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        fourth_table3 = load_CNN15k.clasification_rep()[selected]













if select_clas == "Chat gpt 15k:computer:":
    st.write("# AMLO CLASIFIER")
    st.write("### Number of clasification")
    with st.spinner("Loadig"):
        st.bar_chart(df3["classification_spanish"].value_counts(), color="#4A4646")
    with st.spinner("Loading"):
        st.image("word_cloud3.png", use_column_width=True)

    st.write("### SVC with 15k")
    with st.spinner("Loading table"):
        st.dataframe(third_table3, hide_index=True, use_container_width=True)
        text2 = st.text_input(
            "Input text to clasify with SVG 2",
            label_visibility="visible",
            placeholder="Input texto to clasify ",
            key="input999",
        )
        if st.button("Enviar", key="button999"):
            if text2 != "":
                proba = svg2.predict_text(text2)
                st.write(svg2.predict(proba))
    
    st.write("### CNN with 15k")
    with st.spinner("Loading table"):
        st.dataframe(fourth_table3, hide_index=True, use_container_width=True)
        text3 = st.text_input(
            "Input text to clasify with CNN 2",
            label_visibility="visible",
            placeholder="Input texto to clasify ",
            key="input88",
        )
        if st.button("Enviar", key="button88"):
            if text3 != "":
                proba = load_CNN15k.predict_text(text3)
                st.write(load_CNN15k.predict(proba))

