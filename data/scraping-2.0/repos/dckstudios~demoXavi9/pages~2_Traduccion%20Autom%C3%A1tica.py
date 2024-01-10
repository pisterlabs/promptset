import os
import streamlit as st
import openai

# Configura la clave de API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_text(text, target_language, tone):
    # Define el modelo de lenguaje y el parámetro de temperatura para controlar la diversidad de las respuestas
    model = "text-davinci-003"
    temperature = 0.5

    # Define el tono de la respuesta según la selección del usuario
    if tone == "cercano":
        tone_prompt = "Tono: cercano"
    elif tone == "empresarial":
        tone_prompt = "Tono: Orientado a la venta te habla de usted y te quiere vender lo traducido"
    elif tone == "aspero":
        tone_prompt = "Tono: lejano y serio , cortante y casi mal educado"
    elif tone == "muy latino y cercano":
        tone_prompt = "Tono: muy cercano estilo medellin , parse , papasito , mor , que chimba"
    elif tone == "divertido en plan indio":
        tone_prompt = "Tono: divertido en plan indio de peliculas , yo llmar pies negros"
    else:
        tone_prompt = ""

    # Combinar el texto de entrada y el tono en una sola cadena
    prompt = f"{text}\n\nIdioma objetivo: {target_language}\n{tone_prompt}"

    # Llamar a la API de OpenAI para obtener la traducción
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=100,
        temperature=temperature,
        n=1,
        stop=None
    )

    # Extraer el texto traducido de la respuesta de la API
    translated_text = response.choices[0].text.strip()

    return translated_text


    """
        Función para manejar la lógica principal de la aplicación de traducción.
    
        Esta función muestra una interfaz de usuario utilizando la biblioteca Streamlit para:
        - Obtener el texto a traducir del usuario.
        - Obtener el idioma objetivo para la traducción.
        - Obtener el tono para la traducción.
        - Traducir el texto utilizando la función translate_text().
        - Mostrar el texto traducido.
    
        Parámetros:
            Ninguno
    
        Retorna:
            Ninguno
        """
    
def main():
    st.title("Traductor Automático con Tono")

    # Obtener el texto de entrada del usuario
    text = st.text_area("Introduce el texto a traducir", height=200)

    # Obtener el idioma objetivo de la traducción
    target_language = st.selectbox("Selecciona el idioma objetivo", ["Español", "Francés", "Alemán"])

    # Obtener el tono de la traducción
    tone = st.selectbox("Selecciona el tono", ["Cercano", "Empresarial", "Marketing", "Muy Formal", "Divertido en plan indio"])

    # Mapear el idioma objetivo a su código ISO correspondiente
    if target_language == "Español":
        target_language_code = "es"
    elif target_language == "Francés":
        target_language_code = "fr"
    elif target_language == "Alemán":
        target_language_code = "de"

    # Traducir el texto y mostrar el resultado
    if st.button("Traducir"):
        with st.spinner("Traduciendo..."):
            translated_text = translate_text(text, target_language_code, tone.lower())

        st.success("Texto traducido:")
        st.write(translated_text)

if __name__ == "__main__":
    main()
