import streamlit as st
import pandas as pd
import folium
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from fuzzywuzzy import process
import re
from streamlit_folium import folium_static

import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.header("Configuration")

#ruta_archivo = r'C:\Users\baby_\OneDrive\Escritorio\streamlitproyect\restaurante_google.csv'
df_restaurantes = pd.read_csv("SPOILER_nombres_restaurantes_latitud_longitud.csv")


agent = create_csv_agent(
    ChatOpenAI(temperature=1, model="gpt-4"),
    "SPOILER_Restaurants_Reviews_merged.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)


st.title('Recomendador de Restaurantes en California')


st.subheader('Mapa de California')
coord_california = [36.7783, -119.4179]
m = folium.Map(location=coord_california, zoom_start=6)
st.write(m)


query = st.text_input("Escribe aquí tu pregunta")

if st.button('Buscar'):
    with st.spinner('Procesando...'):
        
        results = agent.run(query)
        
        
        st.subheader('Resultados:')
        st.write(results)
        
        
        if isinstance(results, str):
            # Extrae los nombres de los restaurantes de la respuesta
            names = re.findall(r'(\w+)', results)
            
            # Busca los nombres de los restaurantes en el DataFrame utilizando búsqueda difusa
            matching_restaurants = []
            for name in names:
                match = process.extractOne(name, df_restaurantes['name'])
                if match[1] > 89.5:  # Si el puntaje de coincidencia es mayor a 80
                    matching_restaurants.append(df_restaurantes[df_restaurantes['name'] == match[0]])
            
            matching_restaurants = pd.concat(matching_restaurants)
            
            if not matching_restaurants.empty:
                
                st.subheader('Mapa de Restaurantes')
                m = folium.Map(location=[matching_restaurants['latitude'].mean(), matching_restaurants['longitude'].mean()], zoom_start=12)
                
                for index, row in matching_restaurants.iterrows():
                    folium.Marker([row['latitude'], row['longitude']], popup=row['name']).add_to(m)
                
                # Asegúrate de que el mapa se muestre en Streamlit
                folium_static(m)

