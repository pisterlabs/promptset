import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
import os
import pandas as pd

with st.sidebar:
    api_key_file = st.file_uploader('Aqui va tu key',
                                    type =['txt'])
    if api_key_file is not None:
        key = str(api_key_file.readline().decode("utf-8"))
        os.environ['OPENAI_API_KEY'] = key
        openai.api_key_path = "/Users/lorenzoreinoso/Library/CloudStorage/GoogleDrive-0212511@up.edu.mx/My Drive/Universidad/Semestre 5/PLN/Parcial_3/streamlit/loloskey.txt"
        llm = ChatOpenAI(model_name = 'gpt-3.5-turbo')
        df = pd.read_csv(r'DisneylandReviews.csv', encoding='ISO-8859-1')
        def sentiment(score):
            if score > 3:
                return 'Positive'
            elif score == 3:
                return 'Neutral'
            else:
                return 'Negative'

        # Aplica la función 'sentiment' a la columna 'Rating' para obtener 'Sentiment'
        df['Sentiment'] = df['Rating'].apply(sentiment)

        # Filtra los DataFrames por sentimiento
        df_positivo = df[df['Sentiment'] == 'Positive']
        df_neu = df[df['Sentiment'] == 'Neutral']
        df_neg = df[df['Sentiment'] == 'Negative']

        # Concatena los textos para resumir
        positivo = ' '.join(df_positivo['Review_Text'].tolist())
        negativo = ' '.join(df_neg['Review_Text'].tolist())
        neutro = ' '.join(df_neu['Review_Text'].tolist())

        # Realiza las consultas a GPT-3.5-turbo para obtener los resúmenes
        respuesta_pos = llm.generate(positivo, max_tokens=150)
        respuesta_neu = llm.generate(neutro, max_tokens=150)
        respuesta_neg = llm.generate(negativo, max_tokens=150)

        # Muestra los resúmenes en Streamlit
        st.write('# Resumen Positivo')
        st.write(respuesta_pos)
        st.write('# Resumen Neutro')
        st.write(respuesta_neu)
        st.write('# Resumen Negativo')
        st.write(respuesta_neg)