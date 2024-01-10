import streamlit as st
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SilvIA", page_icon="images/silvIA.png")
st.image("images/silvIA.png", width=100)
st.title('SilvIA')
st.subheader('A sua assistente para gerar questões utilizando Inteligência Artificial')

disciplines = [
               'Língua Portuguesa',
                'Matemática',
                'Biologia',
                'Física',
                'Química',
                'História',
                'Geografia',
                'Sociologia',
                'Filosofia'
              ]
grades = [
          '6º ano do Ensino Fundamental',
          '7º ano do Ensino Fundamental',
          '8º ano do Ensino Fundamental',
          '9º ano do Ensino Fundamental',
          '1ª série do Ensino Médio',
          '2ª série do Ensino Médio',
          '3º série do Ensino Médio'
        ]

openai_api_key = os.getenv('OPENAI_API_KEY')
discipline = st.selectbox('Escolha uma disciplina:', disciplines)
grade = st.selectbox('Escolha uma série:', grades)
content = st.text_input('Conteúdo:')
quantity = st.number_input('Quantidade de questões:', min_value=1, max_value=10, value=1)
multiple_choice = st.checkbox('Com questões de múltipla escolha?')
problem_situation = st.checkbox('Com situação problema?')
competition = st.checkbox('Com questões de concurso?')
answer = st.checkbox('Com resposta no final?')

def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, max_tokens=2048)
  st.info(llm(input_text))

with st.form('my_form'):
  text = 'Crie uma prova para que eu possa estudar e contenha as seguintes características:\n'
  text += f'- Disciplina: {discipline}\n'
  text += f'- Série: {grade}\n'
  text += f'- Conteúdo: {content}\n'
  text += f'- Quantidade de questões: {quantity}\n'
  text += f'- Questões de múltipla escolha: {multiple_choice}\n'
  text += f'- Situação problema: {problem_situation}\n'
  text += f'- Com as repostas/gabarito no final: {answer}\n'
  text += f'- Questões de bancas de concurso público: {competition} e a descrição no início do enunciado quando for o caso\n'

  submitted = st.form_submit_button('Solicitar Questões')
  if submitted:
    generate_response(text)
