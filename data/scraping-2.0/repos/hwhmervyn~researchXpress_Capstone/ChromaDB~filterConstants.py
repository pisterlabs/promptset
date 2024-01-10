from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import OutputFixingParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
import os 
os.environ['LANGCHAIN_TRACING'] = 'false'
from llmConstants import llm

### This script handles the creation of the prompt template and Json Output parsers used in filterExcel #################################

# System template that stays consistent across all questions and excel data input from the user. 
# Instructs LLM not to come up with their own answer from scratch
system_template = '''
Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible.
'''
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human template that changes depending on the question and excel data input
# Format instructions are placed at the end for best practice
human_template = '''
Context:
    Title: {title}
    Abstract: {abstract}

Question: {question}\n
{format_instructions}
'''
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Create the format instructions along with the corresponding Json Output parser
response_schemas = [
  ResponseSchema(name="answer", description="a Yes, No or Unsure answer", type="string"),
  ResponseSchema(name="explanation", description="Explanation on why it is or it isn't relevant", type="string")
]
excel_parser = StructuredOutputParser.from_response_schemas(response_schemas)

chat_prompt = ChatPromptTemplate(
    messages=[system_prompt, human_prompt], 
    partial_variables={
        "format_instructions":excel_parser.get_format_instructions(),
    }
)
# Create an output fixer used when the intial attempt to format output fails. Output fixer detects specifically JsonDecodeError
output_fixer = OutputFixingParser.from_llm(parser=excel_parser, llm=llm)

retry_message = """
[INST]
  <<SYS>>
  The output has format JSONDecodeError preventing it from being parsed into a json object. Please help to correct it.
  Error: 
  {error}
  Current Output:
  {output}
  <</SYS>>
[/INST]
"""
retry_prompt = PromptTemplate.from_template(retry_message)