import openai
import gradio as gr
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import pathlib
import sys
import csv
from datetime import datetime
_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
from scripts.config import Config

# Configurar la API key de OpenAI
cfg = Config()
openai.api_key = cfg.openai_api_key # Reemplaza "YOUR_API_KEY" con tu propia API key
chatgpt= ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
txt_file_path = "req.txt" 

def chat(text):
    prompt_temp_sistema = PromptTemplate(
        template = """Eres un QA Manual crea 5 casos *Críticos* de pruebas para este: {requerimiento} \n\n
                    ###Los casos de prueba deben tener el siguiente formato:
                    ID: \n Descripcion: \n Resultado esperado                 
                    """,
        input_variables=["requerimiento"]
    )

    template_sistema = SystemMessagePromptTemplate(prompt=prompt_temp_sistema)

    #Armamos el template para el humano
    #prompt_temp_humano = PromptTemplate(template="{texto}", input_variables=["texto"])
    #template_humano = HumanMessagePromptTemplate(prompt=prompt_temp_humano)

    chat_prompt = ChatPromptTemplate.from_messages([template_sistema])
    chat_prompt_value = chat_prompt.format_prompt(requerimiento=text).to_messages()
    chat_resp = chatgpt(chat_prompt_value)

    return chat_resp.content


with gr.Blocks(title="Generador de Casos de pruebas", theme="soft", css="syles.css") as demo:
 gr.Markdown(
 """
 # Hola! \n Te ayudaré a generar casos de prueba sobre un requerimiento!
 Escribe un requerimiento debajo: \n
 Por ejemplo: *Pagina de login con 3 campos y un boton de debloquear cuenta* \n\n
 """)
 inp = gr.Textbox(label="Requerimiento",placeholder = "Escribe aquí el requerimiento que debes testear")
 out = gr.Textbox(label="Casos de pruebas")
 
 inp.change(fn = chat, inputs = inp, outputs = out)
 
 demo.launch()