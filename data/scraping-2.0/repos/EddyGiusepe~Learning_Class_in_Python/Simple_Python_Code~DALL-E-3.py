#!/usr/bin/env python3
"""
Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Link de estudo --> https://platform.openai.com/docs/guides/images/usage?context=node

Image Generation: DALL-E 3
==========================
Por padrão, as imagens são geradas com standard qualidade, mas ao 
usar DALL·E 3 você pode definir quality: "hd" para detalhes aprimorados.
Imagens quadradas e de qualidade padrão são as mais rápidas de gerar.

"""
from openai import OpenAI


# Substitua sua chave de API OpenAI:
import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
#openai.api_key  = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

response = client.images.generate(
    model="dall-e-3",
    prompt="Machu Picchu futurístico",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
print(image_url)