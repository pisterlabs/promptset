from langchain.document_loaders import WikipediaLoader
import urllib.request
import streamlit as st
import openai
import requests
import os
import re
import urllib.parse
from bs4 import BeautifulSoup


# Carga el entorno y variables necesarias

openai_api_key = st.secrets["OPENAI_API_KEY"]



def adapt_content_for_kids(query, max_docs=3, max_length=300):
    loader = WikipediaLoader(query=query, load_max_docs=max_docs)
    docs = loader.load()
    docs_text = '\n\n'.join([BeautifulSoup(doc.page_content, features="lxml").text[:max_length] for doc in docs])
    prompt = (
    f"Hello, I'm a friendly robot teacher here to explain '{query}' in a very short, super fun, and easy-to-understand way for kids aged 8 to 12. "
    f"I'll use fun examples, interesting facts, and maybe even a few jokes to make it super entertaining. "
    f"Get ready to learn and laugh at the same time. Oh, and of course, everything I'll tell you will be safe and suitable for your age. Let's go!\n\n"
    f"Limit the response to approximately {max_length} characters to keep it concise.👍"
)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1500
    )
    adapted_content = response.choices[0].message['content'][:1000]
    return adapted_content

"""
Generador de imagen con DALL-E lamentablemente no funciona como esperaba

def generate_image_with_dalle(dalle_prompt):
    try:
        # Enriquecer el prompt para la generación de imágenes
        enhanced_prompt = f"Una imagen para niños que ilustra: {dalle_prompt}. Estilo educativo, adecuado para niños de 6 a 15 años."

        
        response = openai.Image.create(
            prompt=enhanced_prompt,
            n=1,  # número de imágenes a generar
            size="1024x1024"  # tamaño de la imagen
        )

        # La API devuelve una lista de datos de imágenes, extraemos la URL de la primera
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

def is_valid_url(url):
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def clean_text_for_url(text):
    # Utilizamos una expresión regular para eliminar caracteres no válidos en la URL
    cleaned_text = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    return cleaned_text

def save_image_from_url(url, filename):
    try:
        if is_valid_url(url):
            urllib.request.urlretrieve(url, filename)
            return filename
        return None
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None"""