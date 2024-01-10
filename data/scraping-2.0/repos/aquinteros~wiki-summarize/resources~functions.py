import pandas as pd
import wikipediaapi
import openai
import json
# import requests
import streamlit as st
import time
from streamlit_extras.colored_header import colored_header
from streamlit_extras.buy_me_a_coffee import button as buy_me_a_coffee
from streamlit_extras.mention import mention
from urllib.parse import urlparse

st.set_page_config(page_title="Wiki Summary", page_icon="📚", initial_sidebar_state="expanded")

categorias = pd.read_csv('resources/cat.csv', header=None, index_col=0).index.tolist()

@st.cache_resource()
def set_openai_api_key(api_key: str):
    try:
        st.session_state["OPENAI_API_KEY"] = api_key
        openai.api_key = st.session_state["OPENAI_API_KEY"]
        models = pd.json_normalize(openai.Engine.list(), record_path=['data'])
        model_list = models[(models['owner'] == 'openai') & (models['ready'] == True) & (models['id'].str.contains('gpt'))].id
        return model_list
    except Exception as e:
        st.error(e)
        return []
    
def get_completion(prompt, model="gpt-3.5-turbo", temperature=0, num_retries=5, sleep_time=10):
    """function to return content from the openai api prompt"""
    messages = [{"role": "user", "content": prompt}]
    for i in range(num_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            break
        except Exception as e:
            print(f"Retry {i+1}/{num_retries} failed. Error: {e}")
            time.sleep(sleep_time)
		
    return response.choices[0].message["content"]

def import_wiki_page(page_name, language):
    """Importa una página de wikipedia, dado el nombre de la página y el idioma del artículo"""
    headers = {'User-Agent': 'wiki-summarize/0.0 (https://wiki-summarize.streamlit.app/; alvaro.quinteros.a@gmail.com)'}
    wiki = wikipediaapi.Wikipedia(language, headers=headers)
    page = wiki.page(page_name)
    exists = page.exists()
    summary = page.summary
    url = page.fullurl
    sections = page.sections
    return page_name, exists, summary, url, sections

def get_summary(page_name, summary, model, language):
    """Trae un resumen del resumen de una página de wikipedia dada el nombre de la página, el texto del resumen y el idioma del artículo"""

    prompt = f"""
    Tu tarea es generar un resumen corto de un Artículo de wikipedia sobre {page_name} delimitado en triple comillas simples en no más de 40 palabras
    Conserva el tono informativo e impersonal del artículo.
    Omite información de poca relevancia.
    Clasifíca el artículo en una de las siguientes categorías: {categorias}.
    Deriva una lista de como máximo 5 keywords principales del artículo. Evita el nombre del artículo como keyword.
    El idioma del output debe ser '{language}' que es el mismo idioma del artículo.
    El formato de salida SIEMPRE debe ser JSON con los siguientes valores de llave:	[summary, category, keywords].
    Artículo: '''{summary}'''
    """
    
    if len(prompt) > 20000:
        prompt = prompt[:20000] + "'''"
    
    response = json.loads(get_completion(prompt, model).replace('==', '').replace('$ ', '$').replace('# ', '#'))

    return response['summary'], response['category'], response['keywords']

def get_section_summary(page_name, section, model, language):
    """Trae summary de una sección de un artículo en wikipedia, dado el nombre de la página, el texto de la sección y el idioma del artículo"""
    
    prompt = f"""
    Tu tarea es generar un resumen corto de una sección de un Artículo de wikipedia sobre {page_name} delimitada en triple comillas simples en no más de 40 palabras
    Conserva el tono informativo e impersonal de la sección.
    Omite información de poca relevancia, no incluyas información de otras secciones.
    El formato de salida debe ser texto plano en el idioma '{language}' que es el mismo idioma del artículo.
    Artículo: '''{section}'''
    """
    
    if len(prompt) > 20000:
        prompt = prompt[:20000] + "'''"

    response = get_completion(prompt, model).replace('==', '').replace('$ ', '$').replace('# ', '#')

    return response

def return_summary(page_name, model, progress, language):
    """Trae un resumen de una página de wikipedia, dado el nombre de la página y el idioma del artículo"""
     
    page_name, exists, summary, url, sections = import_wiki_page(page_name, language)

    if exists:
        summary, category, keywords = get_summary(page_name, summary, model, language)

        full_text = ''
        
        full_text += '# Summary' + '\n'

        full_text += summary + '\n'

        full_text += '# Category' + '\n'
                
        full_text += category + '\n'

        full_text += '# Keywords' + '\n'

        full_text += ', '.join(keywords) + '\n'

        full_text += '# URL' + '\n'

        full_text += '<' + url + '>' + '\n'

        excluded_titles = ['Referencias', 'Véase también', 'Enlaces externos', 'Fuentes', 'Notas', 'Bibliografía', 'Notes', 'References', 'External links', 'See also', 'Further reading' ,'Sources']

        full_text += '# Sections' + '\n'

        for section in sections:
            if section.title not in excluded_titles:
                full_text += '## ' + section.title + '\n'
                full_text += get_section_summary(page_name, section.full_text, model, language) + '\n'
            progress.progress(sections.index(section)/len(sections))

        full_text += '\n' + '``imported from wikipedia and summarized by openai using <https://wiki-summarize.streamlit.app/>``'
        
        return full_text
    else:
        return "The page doesn't exist"