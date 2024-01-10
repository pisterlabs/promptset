import openai as op
import streamlit as st
import requests as rq
import os
import base64
from io import StringIO

# -------------------------------------------------------------------------------------------------------------------
# Streamlit settings
st.set_page_config(page_title='Mejora imagen', initial_sidebar_state="collapsed", page_icon='🔴', layout="wide")
# -------------------------------------------------------------------------------------------------------------------
op.api_key = "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"

st.title("Mejorando imagenes")
img_mejora=st.file_uploader('Seleciona tu imagen a mejorar por favor')
st.write("***")

btn = st.button("Enviar solicitud")

if btn:
    respuesta = op.Image.create_variation(
        image=img_mejora,
        n=1,
        size='512x512',
        response_format="b64_json"
    )
    
    st.subheader('Mejorando la imagen')
    
    img=respuesta['data'][0]['b64_json']
    img2=base64.b64decode(img)
    st.image(img2)
    
    st.download_button(
        "Descargar imagen",
        data=img2,
        file_name="img.png",
        mime="image/png"
    )
# ----------------------------------------------------------------------------------------------------------------------
st.sidebar.header("Acerca de la App")
st.sidebar.markdown("**Creado por:**")
st.sidebar.write("Nicolas Steven Gutierrez Castiyejo")
st.sidebar.write("nicolasgutierrez029@gmail.com.")
st.sidebar.markdown("**Creado el:** 15/05/2023")