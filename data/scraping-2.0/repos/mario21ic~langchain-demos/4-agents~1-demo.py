import os

from langchain.tools import BaseTool
from typing import Union

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool



class AreaTriangulo(BaseTool):
    name = "Calcular area de un triangulo"
    description = "Use this tool when you need to calculate a triangle area"

    def _run(self, base: Union[int, float], altura: Union[int, float]):
        return (1/2) * float(base) * float(altura)
    
    def _arun(self, base: Union[int, float], altura: Union[int, float]):
        return NotImplementedError("No utiliza async/await")


def multiplier(a: float, b: float) -> float:
    """Multiply the provided floats."""
    return a * b   


API = os.environ['OPENAI_API_KEY']


llm = ChatOpenAI(
    openai_api_key=API,
    temperature=0, # no necesitamos creatividad
    model_name='gpt-3.5-turbo'
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5, # recordar los ultimos 5 mensajes
    return_messages=True
)


tools = [AreaTriangulo()]
# tools = StructuredTool.from_function(multiplier)



# Iniciamos agente
# agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3, # para evitar loops infinitos
#     early_stopping_method='generate',
#     memory=conversational_memory
# )
# print(agent("Calcula el area de un triangulo cuya base es 12.5cm y la altura 6.3 cm"))



import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

hf_model = "Salesforce/blip-image-captioning-large"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = BlipProcessor.from_pretrained(hf_model)
model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)

import requests
from PIL import Image


# Probando un texto a partir de la imagen sin agents
img_url = "https://www.moto1pro.com/sites/default/files/fotosprincipales/novedades_23.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
inputs = processor(image, return_tensors='pt').to(device)
out = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(out[0], skip_special_tokens=True))


from langchain.tools import BaseTool
from langchain.agents import initialize_agent

desc = (
    "use this tool when given the URL of an image that you'd like to be"
    "described. It will return a simple caption describing the image."
)

class ImageCaptionTool(BaseTool):
  name = "Image captioner"
  description = desc

  def _run(self, url: str):
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

  def _arun(self, query: str):
    raise NotIMplementedError("This tool does not support async")

tools = [ImageCaptionTool()]
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

# from PIL import Image


# Mohamed
img_url = "https://phantom-elmundo.unidadeditorial.es/3aa2f112f65eb6b8d7d74cfeb36e3612/resize/473/f/webp/assets/multimedia/imagenes/2021/06/03/16226736385123.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
agent(f"Que es esta imagen?\n{img_url}")

# Black hole
img_url = 'https://news.uchicago.edu/sites/default/files/styles/explainer_hero/public/images/2022-10/sgr%20A%2A%20ESO%20and%20M.%20Kornmesser%20690.jpg?h=06d036b4&itok=Lr5t57tH'
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
agent(f"Que es esta imagen?\n{img_url}")