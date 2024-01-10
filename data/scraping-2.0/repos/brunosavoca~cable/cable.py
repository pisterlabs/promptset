import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai

# Function to extract text from URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Function to paraphrase text using OpenAI
def paraphrase_text(text, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "You are a professional journalist writing for Reuters."
            },
            {
                "role": "user",
                "content": f"Paraphrase in Spanish using a similar amount of words that this given {text}\
                Use transitional phrases, interjections, dangling modifiers, and colloquialisms to create a natural style,\
                and avoid repetitive phrases and unnatural sentence structures. \
                The writing style should be like that of The New York Times.\
                Stick only to the information contained in the source texts without\
                resourcing to your database or any other source, additions, or inventions.\
                Meticulously check names and details when paraphrasing to avoid inaccuracies.\
                Avoid repeating words or ideas, and avoid qualifying adjectives.\
                When quoting textually, add the corresponding attribution."
            }
        ],
    )
    paraphrased_text = response['choices'][0]['message']['content']
    return paraphrased_text

# Streamlit UI

tab1, tab2 = st.tabs(["Comenzar con una URL", "Comenzar con un texto de referencia"])

with st.sidebar:
    st.header("Configuración de clave")
    openai_api_key = st.text_input('Ingresa la clave que recibiste:')

with tab1:
    st.header("Comenzar con una URL")
    url_input = st.text_input('Ingrese la URL:')
    if st.button('Cargar URL'):
        text = extract_text_from_url(url_input)
        st.text_area("Texto original:", value=text, height=200)
        st.session_state.text_to_paraphrase = text

    if st.button('Reinterpretar URL'):
        if 'text_to_paraphrase' in st.session_state:
            with st.spinner('Procesando...'):
                paraphrased_text = paraphrase_text(st.session_state.text_to_paraphrase, openai_api_key)
                st.text_area("Texto reinterpretado:", value=paraphrased_text, height=200)
        else:
            st.warning('Por favor, primero haga clic en "Cargar URL" antes de reinterpretar.')

with tab2:
    st.header('Comenzar con un texto de referencia')
    text_input = st.text_area('Ingrese su texto de referencia:', height=200)
    if st.button('Reinterpretar Texto'):
        with st.spinner('Procesando...'):
            paraphrased_text = paraphrase_text(text_input, openai_api_key)
            st.text_area("Texto reinterpretado:", value=paraphrased_text, height=200)
