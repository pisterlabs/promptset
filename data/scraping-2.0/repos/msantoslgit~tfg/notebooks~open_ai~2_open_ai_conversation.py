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
# # Open AI conversation

# %% [markdown]
# ## Parte necesaria de "open_ai_conection.ipynb"

# %%
# from openai import OpenAI

import sys
from pathlib import Path 
path_to_tfg = Path().absolute().parent.parent
sys.path.insert(0, str(path_to_tfg))

from component import API_KEY

client = OpenAI(api_key=API_KEY)
model = "gpt-3.5-turbo"
max_tokens = 50

conversation = [{"role": "system",                             #Asignamos como rol al sitema
                 "content": "You are a helpful assistant."}]   #Definimos que es un ayudante de utilidad 


# %% [markdown]
# # Conversación con la IA

# %% [markdown]
# - Primero importamos **get_response** 
# - Los parámetros necesarios son: client, model, conversation, content, max_tokens
# - Nos retorna la respuesta del sistema al último contentresponse
#   

# %%
# def get_response(client, model, conversation, content, max_tokens):
#     conversation.append({"role": "user", "content": content})

#     completion = client.chat.completions.create(
#         model=model,
#         messages=conversation,
#         max_tokens=max_tokens
#     )
#     assistant_response = completion.choices[0].message.content

#     return assistant_response

# %%
from old.openAI_functions import get_response
content = "Hola chat me llamo Mario"

response = get_response(client, model, conversation, content, max_tokens)
print(response)

# %% [markdown]
# - Como vemos la primera parte funciona correctamente
# - Ahora vamos a preguntarle si recuerda como me llamo 

# %%
content = "¿Hey chat recuerdas como me llamo?"
response = get_response(client, model, conversation, content, max_tokens)
print(response)

# %% [markdown]
# - El chat ha recordado mi nombre ya que en la función de get_response se van añadiendo las preguntas a la conversación
# - conversation.append({"role": "user", "content": content})
# - Lo que falta ahora es que el chat recuerde sus respuestas
# - Para esto hemos hecho la función **add_context_response** 

# %%
# def add_context_response(conversation, response):
#     conversation.append({"role": "assistant", "content": response})

# %%
from old.openAI_functions import add_context_response

add_context_response(conversation, response)

# %% [markdown]
# ## Funcionamiento 
# - Establecemos un bucle, salimos con "exit"
# - Mientras le hablamos a la IA va guardando en la conversación nuestras preguntas y sus respuestas

# %%
print("¿Sobre que quieres hablar? ")
while True:

    content = input(" ")

    if content == "exit":
        break

    response = get_response(client, model, conversation, content, max_tokens)

    print(response)
    add_context_response(conversation, response)

# %% [markdown]
# ## Cerrar sesión

# %% [markdown]
# - Código para cerrar la sesión

# %%
# def close_session(client, model, max_tokens):
#     # Agrega un mensaje para cerrar la sesión
#     conversation = [{"role": "system", "content": "Close sesion."}]

#     # Realiza una última llamada para cerrar la sesión
#     completion = client.chat.completions.create(
#         model=model,
#         messages=conversation,
#         max_tokens=max_tokens
#     )
#     # assistant_response = completion.choices[0].message['content']
#     assistant_response = completion.choices[0].message.content
#     print("Session closed")

# %%
from old.openAI_functions import close_session

close_session(client, model, conversation, max_tokens)

# %%

# %%
