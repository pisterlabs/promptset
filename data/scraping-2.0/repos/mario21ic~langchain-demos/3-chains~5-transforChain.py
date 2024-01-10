"""
Se recomienda usar para preprocesamiento para enviar o recibir de una LLM
Ejem limpiar textos antes de mandar a otros Chains
"""

import os
from langchain import LLMChain, OpenAI, PromptTemplate

API = os.environ['OPENAI_API_KEY']
llm = OpenAI(openai_api_key=API)

from langchain.chains import TransformChain


def eliminar_brincos(input):
    """Elimina los brincos de línea de un texto."""
    texto = input["texto"]
    return {"texto_limpio": texto.replace("\n", " ")}


cadena_transformacion = TransformChain(input_variables=["texto"],
                                        output_variables=["texto_limpio"],
                                        transform=eliminar_brincos)

prompt = '''\nEste es un texto \ncon brincos de\n línea.\n\n'''


print(cadena_transformacion.run(prompt))