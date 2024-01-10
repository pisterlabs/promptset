"""
Los modelos que utilizamos eran para completar texto, sin embargo podemos tambien usar modelos especificos para chat como ChatGPT
"""

import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import PromptTemplate
from langchain.llms import OpenAI

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate


output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()


OPENAI_API_KEY = os.environ['API_KEY']
MODEL="text-davinci-003"
llm_openai = OpenAI(model_name=MODEL, openai_api_key=OPENAI_API_KEY)

print("format_instructions", format_instructions)

template_basico_parser = """Cuales son los ingredientes para preparar {platillo}\n{como_parsear}"""

prompt_temp_parser = PromptTemplate(input_variables=["platillo"], 
                                    template = template_basico_parser, 
                                    partial_variables={"como_parsear": format_instructions})

# promt_value_parser = prompt_temp_parser.format(platillo="tacos al pastor")
promt_value_parser = prompt_temp_parser.format(platillo="ceviche piurano")
print(promt_value_parser)


respuesta = llm_openai(promt_value_parser)
print(respuesta)

print(output_parser.parse(respuesta))