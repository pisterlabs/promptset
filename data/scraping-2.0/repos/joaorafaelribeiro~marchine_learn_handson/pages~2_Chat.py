import time
import streamlit as st
import pandas as pd
from io import StringIO
from streamlit_modal import Modal
import datetime
from util.pdf import generate_pdf
import os
from openai import OpenAI


flai = 'recursos/logo flai com sombra.png'

st.title("🤖 Assistente Pessoal")
    
chave = st.sidebar.text_input('Chave da API OpenAI', type = 'password',value='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')
temperature = st.sidebar.slider('Criatividade',min_value=.8,max_value=1.5,value=.1,step=.1)
max_tokens = st.sidebar.number_input('Tamanho',min_value=100,max_value=200,value=100,step=1)
estilo = st.sidebar.radio('Estilo',options=['Inteligente','Sarcástico','Extrovertido'],index=0)
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("RAG (formato TXT)",accept_multiple_files=False)
contexto = ''
if uploaded_file is not None:

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    contexto = stringio.read()
st.sidebar.divider()
salvar = st.sidebar.button('Finalizar Conversa',type='primary',use_container_width=True)


modal = Modal("Histórico",key='modal',padding=20,max_width=600)

if salvar:
    modal.open()

client = OpenAI(api_key=chave)



def perigo(mensagem):
  response = client.moderations.create(input=mensagem)
  scores = response.results[0].category_scores
  return  scores.violence > .75 or scores.sexual > .75 or scores.hate > .75 or scores.harassment > .75

config = f'''Você é um assistente pessoal seja {"Sarcastico" if estilo == 1 else "Inteligente" if estilo == 0 else "Extrovertido"}. Para responder, considere também as seguintes informações: \n\n{contexto}'''

# Iniciar Historico Chat
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [{"role": 'system', "content": config}]
else:
    st.session_state.mensagens[0] = {"role": 'system', "content": config}

# Aparecer o Historico do Chat na tela
for mensagens in st.session_state.mensagens[1:]:
    with st.chat_message(mensagens["role"]):
        st.markdown(mensagens["content"])


# React to user input
prompt = st.chat_input("Digite alguma coisa")

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.mensagens.append({"role": "user", "content": prompt})

    resposta = ""

    if not perigo(prompt):
        chamada = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            max_tokens =  max_tokens,
            temperature = temperature,
            messages = st.session_state.mensagens
        )
        resposta = chamada.choices[0].message.content
    else:
        resposta = 'Pergunta inadequada'

    # Display assistant response in chat message container
    with st.chat_message("system"):
        st.markdown(resposta)
    # Add assistant response to chat history
    st.session_state.mensagens.append({"role": "system", "content": resposta})

if modal.is_open():
    with modal.container():
        historico = pd.DataFrame({
            'from': [m['role'] for m in st.session_state.mensagens],
            'messages': [m['content'] for m in st.session_state.mensagens]
        })
        st.markdown(f'data: {datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}')
        st.write(historico)
        filename = f'{os.getcwd()}\historico.pdf'
        generate_pdf(historico,filename)
        file = open(filename, 'rb')
        #pdfkit.from_string(historico.to_html())
        st.download_button('Download',data=file.read(),file_name='historico.pdf',mime='application/pdf')
        file.close()
        st.session_state.mensagens = [{"role": 'system', "content": config}]
