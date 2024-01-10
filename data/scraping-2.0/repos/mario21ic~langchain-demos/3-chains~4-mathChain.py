"""
Genera un code python para luego evaluarlo
"""

import os
from langchain import OpenAI, LLMMathChain

API = os.environ['OPENAI_API_KEY']
llm = OpenAI(openai_api_key=API)

cadena_mate = LLMMathChain(llm=llm, verbose=True)
print(cadena_mate.run("Cuanto es 432*12-32+32?"))
