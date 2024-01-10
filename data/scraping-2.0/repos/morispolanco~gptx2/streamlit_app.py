import streamlit as st
import openai

# Configurar la clave de la API de OpenAI
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if not api_key:
    st.warning("Please enter a valid API key to continue.")
else:
    openai.api_key = api_key
    # Continuar con el resto del código que utiliza la clave de API

# Título de la aplicación
st.title("Chatbots de GPT-3")

# Ingresar el mensaje inicial
mensaje_inicial = st.text_input("Ingresa tu mensaje inicial")

# Conversación entre los chatbots
def conversacion(mensaje_inicial):
    # Inicializar la conversación con el mensaje inicial
    conversacion = [mensaje_inicial]

    # Bucle para el diálogo entre los chatbots
    while True:
        # Obtener la respuesta del primer chatbot
        respuesta_1 = openai.Completion.create(
            engine="text-davinci-003",
            prompt=conversacion,
            max_tokens=50,
            temperature=0.7
        )

        # Agregar la respuesta del primer chatbot a la conversación
        conversacion.append(respuesta_1.choices[0].text.strip())

        # Obtener la respuesta del segundo chatbot
        respuesta_2 = openai.Completion.create(
            engine="text-davinci-003",
            prompt=conversacion,
            max_tokens=50,
            temperature=0.7
        )

        # Agregar la respuesta del segundo chatbot a la conversación
        conversacion.append(respuesta_2.choices[0].text.strip())

        # Mostrar la conversación en la interfaz de la aplicación
        st.write("Chatbot 1:", conversacion[-2])
        st.write("Chatbot 2:", conversacion[-1])

        # Preguntar al usuario si desea continuar la conversación
        continuar = st.selectbox("Continuar la conversación?", ["Sí", "No"])

        # Terminar la conversación si el usuario elige "No"
        if continuar == "No":
            break

# Iniciar la conversación si se ha ingresado un mensaje inicial
if mensaje_inicial:
    conversacion(mensaje_inicial)
