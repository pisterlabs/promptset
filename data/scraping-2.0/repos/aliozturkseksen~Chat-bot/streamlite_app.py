# st es una clase muy parecida a Tk, por lo que piensa asi en ella.
import streamlit as st
# En general st.session_state es un diccionario.
from streamlit_chat import message
from utils import get_initial_message, get_chatgpt_response, update_chat
import os
from dotenv import load_dotenv
load_dotenv()
import openai

openai.api_key = "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"

st.title("Chatbot : ChatGPT and Streamlit Chat Hector")
st.subheader("AI Tutor:")

# Caja donde se mostrara una caja de seleccion mostrando la opciones
model = st.selectbox(
    # Titulo de caja
    "Select a model",
    # Opciones a elegir
    ("gpt-3.5-turbo", "gpt-4")
)

# En caso de estar generada o expirada la sesion, la declaramos vacia.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Aqui se recoje el texto que entra al chatbot
query = st.text_input("Query: ", key="input")

# Con esto inicialiso los mensajes en la estado de la sesion
if 'messages' not in st.session_state:
    st.session_state['messages'] = get_initial_message()
 
# Si el query no esta vacio
if query:
    # Muestra temporalmente un texto mientras genera un bloque de codigo
    with st.spinner("generating..."):
        # Extraemos el ultimo hilo de la conversacion.
        messages = st.session_state['messages']
        # Actualizamos el hilo de la conversacion para el usario
        messages = update_chat(messages, "user", query)
        
        # st.write("Before  making the API call")
        # st.write(messages)
        
        # Generamos la respuesta del modelo en base al hilo
        response = get_chatgpt_response(messages,model)
        # Actualizamos el hilo de la conversacion para el asistente
        messages = update_chat(messages, "assistant", response)
        # Se incluyen los nuevos mensajes en el estado de la sesion correspondientemente
        st.session_state.past.append(query)
        st.session_state.generated.append(response)
        
if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # If is_user=True, se mostrara el mensaje del lado derecho.
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

    with st.expander("Show Messages"):
        st.write(messages)
