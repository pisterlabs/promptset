import streamlit as st
import os
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_text(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message

st.title("Buscador de Mejores Precios")
product = st.text_input("Ingrese el producto a buscar")
location = "Guatemala" 

if st.button("Buscar"):
    prompt = f"Encuentra el mejor precio en línea para un(a) {product} en {location} y provee el link para comprarlo"   
    response = generate_text(prompt)
    st.success(response)
