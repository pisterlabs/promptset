# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Open AI connection

# %% [markdown]
# - Import libraries 

# %%
from openai import OpenAI

# %% [markdown]
# - Set up project path as sys.path

# %%
import sys
from pathlib import Path 
path_to_tfg = Path().absolute().parent.parent
print(path_to_tfg)
sys.path.insert(0, str(path_to_tfg))

# %% [markdown]
# - Make sure the API_KEY is aviable 

# %%
from component import API_KEY
print(API_KEY)

# %% [markdown]
# - Set up API parameters
# - Needed client, model and max_tokens 
# - Client is the user connection 
# - Model is the engine we would like to ask from OpenAI (GPT-3, GPT-3.5-turbo)
# - max_token will limitate the lenght of the response by the Model

# %%
client = OpenAI(api_key=API_KEY)
model = "gpt-3.5-turbo"
max_tokens = 50

# %% [markdown]
# - Iniciamos la conversación con la API
# - En el set up inicial le asignamos un rol
# - Este rol puede cambiar dependiendo de que queremos que nos responda o en que tarea queremos que nos ayude

# %%
conversation = [{"role": "system",                             #Asignamos como rol al sitema
                 "content": "You are a helpful assistant."}]   #Definimos que es un ayudante de utilidad 

# %%
print(conversation)

# %%
# conversation = [{"role": "system",                             #Asignamos como rol al sitema
#                  "content": "You are an SQL expert"}]          #Definimos que es un experto en SQL 

# %%
# print(conversation)

# %% [markdown]
# - Lo que hacemos ahora es añadir un INPUT a esta conversación 
# - Vamos a leer una pregunta del usuario
# - Esta pregunta la vamos a añadir a la conversación y le vamos a pedir al sistema que responda

# %%
content = input(" ")

# %%
conversation.append({"role": "user", "content": content})     #Añadimos la pregunta a la conversación
 
completion = client.chat.completions.create(                  #Generamos la petición
    model=model,
    messages=conversation,
    max_tokens=max_tokens
)

assistant_response = completion.choices[0].message.content    #Recibimos la respuesta de la IA 

print(assistant_response)
