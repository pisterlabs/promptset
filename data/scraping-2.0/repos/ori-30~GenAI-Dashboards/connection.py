import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components  # Import Streamlit
import requests
import json
from openai import OpenAI, AzureOpenAI
from typing import List
from functions import *
import MetadataExtractor as me
from snowflake.snowpark import Session

st.set_page_config(
    page_title="GenAI domains",
    page_icon=":heart:",
)

if 'display_result' not in st.session_state:
	st.session_state.display_result = True
if 'reset' not in st.session_state:
    st.session_state.reset = False
if 'area' not in st.session_state:
	st.session_state['area']=""
if 'description' not in st.session_state:
	st.session_state['description']=""
if 'prompt_metadata' not in st.session_state:
	st.session_state['prompt_metadata']=""

def callback():
	if des:
		st.session_state['area']=area
		st.session_state['description']=des
		st.session_state.display_result=False
		st.session_state.reset=False
		s.session_state.prompt_metadata=prompt_metadata
	else:
		st.error("Por favor, rellene ambos campos")

if not st.session_state.display_result:
	metadata = st.session_state["prompt_metadata"]
	promt_json= open('promptjson.txt', 'r').read()
	#abrir openAI key con streamlit secrets
	client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
	st.write(metadata)

	#crear modelo por defecto
	if "openai_model" not in st.session_state:
	    st.session_state["openai_model"] = "gpt-3.5-turbo"

	#inizializar chat
	if "messages" not in st.session_state:
		st.session_state.messages = [{"role": "system", "content": metadata}]
		st.session_state.messages.append({"role": "system", "content": promt_json})
		cl=client.chat.completions.create(model=st.session_state["openai_model"], messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages], stream=True)
		full_response=""
		for response in cl:
			full_response +=(response.choices[0].delta.content or "")
		st.session_state.messages.append({"role": "system", "content": full_response})
		if "domains" not in st.session_state:
			st.session_state["domains"]=full_response

	#creamos la sidebar
	with st.sidebar:
		st.header("Chatbot", divider='rainbow')
		# Aceptamos input del usuario
		prompt = get_text()
		#mostramos el chat de mensajes desde el historial
		for message in st.session_state.messages:
			
			if message["role"]!="system":
			    with st.chat_message(message["role"]):
		        	st.markdown(message["content"])
		if prompt:
			#añadimos mensaje del usuario
			st.session_state.messages.append({"role": "user", "content": prompt})
			#mostramos mensaje del usuario
			with st.chat_message("user"):
				st.markdown(prompt)
			# Display assistant response in chat message container
			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
			cl=client.chat.completions.create(model=st.session_state["openai_model"], messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages], stream=True)
			for response in cl:
				full_response +=(response.choices[0].delta.content or "")
				message_placeholder.markdown(full_response + "▌")
			message_placeholder.markdown(full_response)
			st.session_state.messages.append({"role": "assistant", "content": full_response})

	container= st.container()
	with container:
		bootstrap()
		create_sql_statment(container)
		dominios=get_JSON()
		create_domains(dominios["dominios"], container)

if st.session_state.display_result:
	st.header("Domain GenAI")
	selector=st.radio("Selecciona la API: ",["OpenAI", "AzureOpenAI"])
	if selector == "AzureOpenAI":
		ao_key=st.text_input("Azure api tokne: ",type="password")
		ao_version=st.text_input("Azure api version:")
		ao_endpoint=st.text_input("Azure endopoint:")
		model=st.text_input("Azure deployment name:")

		client = AzureOpenAI(
			
			)
		model="modelo3"
	else:
		openai_input=st.text_input("OpenAi api token: ",type="password")
		model=st.text_input("OpenAi model: ")
		client = OpenAI(
			api_key=openai_input
		)

	st.header("Configuracion Snowflake")

	acc_input=st.text_input("Identificador cuenta de Snowflake","")
	user_input=st.text_input("Nombre de usuario","")
	pass_input=st.text_input("Contraseña","",type='password')

	input3 = st.text_input("Base de datos:", "")

	# Configurar la barra lateral
	st.header("Información de la empresa")
	area=get_area()
	des=get_des()
	prompt_metadata =me.get_metadata(acc_input,user_input,pass_input,input3)
	prompt_metadata += f"\n\nEsta es la descripción de la empresa: {st.session_state.descripcion}\nEstas son las áreas de negocio: {st.session_state.area}"
		
	send=st.button("Generar", disabled=(area is ""), on_click=callback)



