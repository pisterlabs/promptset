import streamlit as st
import openai
from streamlit_javascript import st_javascript
import numpy as np
from textblob import TextBlob

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(page_title = "Emotions Tweets", page_icon="assets/TwitterIcon.png")

API_KEY = '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'
datos_emociones = []


st.markdown("""
<style>
h1, h2, h3, h4, h5, h6 {
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


def get_from_local_storage(k):
    v = st_javascript(f"JSON.parse(localStorage.getItem('{k}'));")
    return v or {}


def traducir_texto(texto, idioma_origen, idioma_destino):
    # Crear un objeto TextBlob con el texto en el idioma de origen
    blob = TextBlob(texto)
    
    # Realizar la traducción al idioma destino
    texto_traducido = blob.translate(from_lang=idioma_origen, to=idioma_destino)
    
    return str(texto_traducido)


def construir_mensaje():
    try:
        data = get_from_local_storage("tendencia")
        lista_frecuencias = get_from_local_storage("porcentajes")
        sorted(lista_frecuencias)
        
        MENSAJE = f"""En estos momentos eres un analista de datos: 
        Dame una conclusión bien estructurada, explicando un contexto desde internet  y 
        argumentando el porque de los datos a partir de la siguiente lista de datos, 
        los cuales son emociones procesadas de diferentes 
        tweets de Twitter que están en la tendencia { str(data[0]) }
        de Colombia en estos momentos (ten en cuenta que los números más cercanos a 1 
        son una emoción positiva y los números más cercanos a -1 son 
        una emoción negativa, por lo que los más cercanos a 0 son neutrales, 
        de -1 a -0.6 es muy enojado, de -0.5 a -0.1 es enojado, si es 0 es neutral, 
        de 0.1 a 0.5 es alegre, y de 0.6 a 1 es muy alegre. no me digas conclusiones con 
        numeros, explica el contexto y dame el porque de los datos): { str(lista_frecuencias) }""" 

        try:
            global datos_emociones
            datos_emociones = lista_frecuencias.copy()
        except NameError as e:
            print("Definiendo Emoción")
        return MENSAJE
    except KeyError as e:
        MENSAJE = "No se ha seleccionado ninguna tendencia. Por favor, ve a la sección de Trending Topics y selecciona una tendencia."
        st.components.v1.html('<img src="https://media2.giphy.com/media/yJil9u57ybQ9movc6E/giphy.gif?cid=ecf05e47s72rt1y7ev7779rz96vk405kbzb0b745slavs56r&rid=giphy.gif" alt="GIF">', width=600, height=300)
        return MENSAJE


def traducir_texto(texto, idioma_origen, idioma_destino):
    # Crear un objeto TextBlob con el texto en el idioma de origen
    blob = TextBlob(texto)
    
    # Realizar la traducción al idioma destino
    texto_traducido = blob.translate(from_lang=idioma_origen, to=idioma_destino)
    
    return str(texto_traducido)


def openai_api(mensaje):
    openai.api_key = API_KEY

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=mensaje,
        max_tokens=3000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    return response.choices[0].text


def emocion_principal(lista):
    if lista != []:
        conteo_emociones = {
            "Muy enojados": 0,
            "Enojados": 0,
            "Neutrales": 0,
            "Alegres": 0,
            "Muy alegres": 0
        }

        for emocion in lista:
            if -1 <= emocion < -0.5:
                conteo_emociones["Muy enojados"] += 1
            elif -0.5 <= emocion <= -0.1:
                conteo_emociones["Enojados"] += 1
            elif -0.1 < emocion < 0.1:
                conteo_emociones["Neutrales"] += 1
            elif 0.1 <= emocion <= 0.5:
                conteo_emociones["Alegres"] += 1
            elif 0.5 < emocion <= 1:
                conteo_emociones["Muy alegres"] += 1

        emocion_principal = max(conteo_emociones, key=conteo_emociones.get)
        return emocion_principal


def conclusiones():
    st.title("Conclusiones:")
    try:
        conc = openai_api(construir_mensaje())
        st.title("Español: ")
        st.write(conc)
        conc_english = traducir_texto(conc, "es", "en")
        st.title("Inglés: ")
        st.write(conc_english)
        if emocion_principal(datos_emociones) == "Muy enojados":
            st.components.v1.html('<img src="https://media.tenor.com/lTe5PjawzRAAAAAC/furia-intensamente.gif" alt="GIF">', width=600, height=300)
            st.components.v1.html('<br>')
            
        elif emocion_principal(datos_emociones) == "Enojados":
            st.components.v1.html('<img src="https://media.tenor.com/WS55YpItsqIAAAAC/enserio-deverdad.gif" alt="GIF">', width=600, height=300)
            st.components.v1.html('<br>')

        elif emocion_principal(datos_emociones) == "Neutrales":
            st.components.v1.html('<img src="https://st1.uvnimg.com/dims4/default/40b28a9/2147483647/thumbnail/480x270/quality/75/format/jpg/?url=https%3A%2F%2Fuvn-brightspot.s3.amazonaws.com%2Fassets%2Fvixes%2Fbtg%2Fgiphy_77_0.gif" alt="GIF">', width=600, height=300)
            st.components.v1.html('<br>')

        elif emocion_principal(datos_emociones) == "Alegres":
            st.components.v1.html('<img src="https://i.pinimg.com/originals/b5/49/f7/b549f732fa170a0cbd6629fd5654c2dd.gif" alt="GIF">', width=600, height=400)
            st.components.v1.html('<br>')

        elif emocion_principal(datos_emociones) == "Muy alegres":
            st.components.v1.html('<img src="https://media.tenor.com/ZUkMEj5iYkAAAAAd/kun-aguero-dance.gif" alt="GIF">', width=600, height=450)
            st.components.v1.html('<br>')
            
        enviar_correo( conc )
        
    except:
        st.write('Nos banearon de OpenAI por hacer tantas peticiones chulo las conclusiones, pero relajate horrible profe! ;D')
        st.components.v1.html('<img src="https://media.tenor.com/ZUkMEj5iYkAAAAAd/kun-aguero-dance.gif" alt="GIF">', width=600, height=450)
        
    

def enviar_correo( info ):
    receiver = st.text_input("Ingrese su correo para recibir las conclusiones")

    # Crear un botón
    clicked = st.button("Enviar correo")

    # Verificar si el botón ha sido presionado
    if clicked:
        enviar( receiver, info  )
    

def enviar( receiver, info ):
    # Configuración del servidor SMTP y credenciales de inicio de sesión
    try:
        msg = MIMEMultipart()
        message = info
        # setup the parameters of the message 
        password = "nryblungwwvhejha"
        msg['From'] = "emotiontweetsteams@gmail.com"
        msg['To'] = receiver
        msg['Subject'] = "Conclusiones de Emotions Tweets"
        # add in the message body 
        msg.attach(MIMEText(message, 'plain'))
        #create server 
        server = smtplib.SMTP('smtp.gmail.com: 587')
        server.starttls()
        # Login Credentials for sending the mail 
        server.login(msg['From'], password)
        # send the message via the server. 
        server.sendmail(msg['From'], msg['To'], msg.as_string())
        server.quit()
    except NameError as a:
        print( a )


conclusiones()