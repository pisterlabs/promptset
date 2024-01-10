import pprint

from pydantic import BaseModel, Field
import configparser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.utils import openai_functions

config = configparser.ConfigParser()
config.read('../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')


class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")


class ProductSearch(BaseModel):
    """Call this with product name to get the price of product"""
    product_name: str = Field(description="name of product to look up")


functions = [
    openai_functions.convert_pydantic_to_openai_function(WeatherSearch),
    openai_functions.convert_pydantic_to_openai_function(ProductSearch)
]

llm = ChatOpenAI(openai_api_key=openai_api_key).bind(functions=functions)
prompt = ChatPromptTemplate.from_template("今天{input}天气怎么样")
chain = prompt | llm

response = chain.invoke({"input": "北京"})
print(response)

args = response.additional_kwargs['function_call']['arguments']
args = eval(args)
print(args)
