### Lbrerias

# Conexiones

from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Manipulación

import pandas as pd
import dotenv
import os
import matplotlib.pyplot as plt 
import seaborn as sns

# Otros

import env
import os

# Streamlit

import streamlit as st
from streamlit_chat import message

### Datos y API

# Cargar datos

df = pd.read_csv("Original\\MarteaAI\\Data\\video_games_sales.csv")

# Conectar API

os.environ["OPENAI_API_KEY"] = env.OPENAI_API_KEY

### Crear y configurar el bot (Anthea)

# Programación de Anthea

Anthea = create_pandas_dataframe_agent(
    ChatOpenAI(temperature = 0,
               model = env.MODEL,
               openai_organization = "org-4K9pPkXLsBYD9znBj07fnte2"),
    df,
    verbose = True,
    agent_type = AgentType.OPENAI_FUNCTIONS)

# Formato personalizado de respuesta de Anthea

formato = """
Data una pregunta del usuario:
1. Crea una consulta de sqlite3.
2. Revisa los resultados e interpretalos.
3. Devuelve el dato.
4. si tienes que hacer alguna aclaración o devolver cualquier texto que sea siempre en español y claro.
#{question}
"""

# Función para hacer la consulta

def consulta(input_usuario):
    consulta = formato.format(question = input_usuario)
    resultado = Anthea.run(consulta)
    return(resultado)

### Generación y diseño del Framework (Opcional)

st.title("Anthea")
st.write("¡Hola! Soy una asistente virtual creada por Marte AI, una Startup del rubro de análisis de datos. Mi objetivo es apoyarte en todo lo que tenga que ver con datos, análisis y similares. Puedes hacerme la duda que sea sobre la base de datos que me haz dado.")

st.code("Esta es una versión Beta. No me pidas demasiado 🥺. Estoy chiquita y llevo poco tiempo entrenando.", language = "python")

if 'preguntas' not in st.session_state:
    st.session_state.preguntas = []
if 'respuestas' not in st.session_state:
    st.session_state.respuestas = []

def click():
    if st.session_state.user != "":
        pregunta = st.session_state.user
        respuesta = consulta(pregunta)
        st.session_state.preguntas.append(pregunta)
        st.session_state.respuestas.append(respuesta)
        st.session_state.user = ''

with st.form('my-form'):
   query = st.text_input('¿En qué te puedo ayudar?:', key='user', help = 'Estoy diseñada para responder cualquier tipo de pregunta referida a los datos que me haz dado.')
   submit_button = st.form_submit_button('Consultar', on_click = click)

if st.session_state.preguntas:
    for i in range(len(st.session_state.respuestas)-1, -1, -1):
        message(st.session_state.respuestas[i], key=str(i))

    continuar_conversacion = st.checkbox('¿Quieres cambiar el tema?')
    if not continuar_conversacion:
        st.session_state.preguntas = []
        st.session_state.respuestas = []
