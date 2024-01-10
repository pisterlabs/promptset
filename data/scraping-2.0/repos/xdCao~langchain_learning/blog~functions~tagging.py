import configparser
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

config = configparser.ConfigParser()
config.read('../../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')


class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")


llm = ChatOpenAI(openai_api_key=openai_api_key)
functions = [
    convert_pydantic_to_openai_function(Tagging)
]
llm = llm.bind(functions=functions, function_call={"name": "Tagging"})

prompts = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

chain = prompts | llm | JsonOutputFunctionsParser()

response = chain.invoke({"input": "Today‘s weather is awful"})
print(response)
