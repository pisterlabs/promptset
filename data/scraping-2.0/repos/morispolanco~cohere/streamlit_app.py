import streamlit as st
import cohere
import json

# Ingresa tu API key de Cohere aquí
API_KEY = 'KDhsUSaH5D01hlEWHEOfZvNhowIGPZC2cUcCP1sO'


co = cohere.Client(API_KEY)

st.title("Leyes de Guatemala")

question = st.text_input("Escribe tu pregunta:")

if st.button("Responder"):

  response = co.chat(
    model='command',
    message=question,  
    chat_history=[{"role": "User", "message": question}]
  )

  st.write("Respuesta:")

  text = ""
  for message in response.stream:
    text += message["message"]

  st.write(text)

  # Para depurar
  data = {
    "message": response.message,
    "stream": [m["message"] for m in response.stream]
  }

  st.write(json.dumps(data))

co = cohere.Client(API_KEY)

st.title("Leyes de Guatemala")

question = st.text_input("Escribe tu pregunta:")

if st.button("Responder"):

  response = co.chat(
    model='command',
    message=question,
    temperature=0.3,
    chat_history=[
      {"role": "User", "message": question}
    ],
    prompt_truncation='AUTO',
    stream=True,
    citation_quality='accurate',
    connectors=[{"id":"web-search"}],
    documents=[]
  )

  st.write("Respuesta:")

  # Convertimos la respuesta a JSON para mostrar
  response_json = json.dumps(response)  

  st.write(response_json)
