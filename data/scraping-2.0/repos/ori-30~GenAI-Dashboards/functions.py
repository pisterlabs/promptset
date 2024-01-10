import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components  # Import Streamlit
import requests
import json
import openai
from typing import List

def get_text():

    input_text = st.text_input("You: ","", key="input")
    return input_text

def get_area():

    input_text = st.text_input("Areas de negocio: ","", key="area")
    return input_text

def get_des():
    input_text = st.text_input("Descripción de la empresa: ","", key="des")
    return input_text 

def create_gpt_completion(ai_model: str, messages: List[dict]) -> dict:
    openai.api_key = st.secrets.api_credentials.api_key
    completion = openai.ChatCompletion.create(
        model=ai_model,
        messages=messages,
    )
    return completion


def get_JSON():
	try:
		dominios = st.session_state.domains
	except:
		st.error("error json")
	return json.loads(dominios)

def tables(alltables):
	r=""
	for table in alltables:
		r+= """<p class="card-text">%s</p>""" % str(table)
	return r
def create_card(title, alltables):

	card="""
		<div class="m-1 p-1"style="padding: 2px 16px;">
			<div class="card m-2" style="width: 18rem;">
			  <div class="card-body bg-light">
			    <h3 class="card-title">%s</h3>
	""" % str(title)
	card+=tables(alltables)

	card+=""" 			  
				</div>
			</div>
		</div>
		"""
	return card

def create_domains(dominios, container):
	c = container.columns(2)
	i=0
	for dominio in dominios:
		d= create_card(dominio["nombre"], dominio["tablas"])
		c[i].markdown(d, unsafe_allow_html= True)
		i=(i+1)%2


def create_sql_statment(container):
	sql="Esto es una sentencia sql"
	box="""
		<div class="card w-100 m-2">
			<div class="card-body w-100 bg-info">
				<p>%s</p>
			</div>
		</div>
	""" % str(sql)
	container.markdown(box, unsafe_allow_html= True)


def bootstrap():
	_bootstrap="""<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">"""
	st.markdown(_bootstrap, unsafe_allow_html= True)