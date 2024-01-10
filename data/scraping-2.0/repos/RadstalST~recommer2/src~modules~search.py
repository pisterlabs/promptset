import json
import os
from typing import Optional
from serpapi import GoogleSearch

from langchain.agents import (AgentType, OpenAIFunctionsAgent, Tool,
                              initialize_agent, load_tools)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import (CommaSeparatedListOutputParser,
                                      PydanticOutputParser, ResponseSchema,
                                      StructuredOutputParser)
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               PromptTemplate)
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.schema import SystemMessage
from langchain.utilities import SerpAPIWrapper, SQLDatabase
from pydantic import BaseModel, Field, validator


class ProductScope(BaseModel):
    desire: str
    tags: list[str]
    description: str = "not available"
class ProductInfo(BaseModel):
    name: str= Field(description="name of the product")
    # link: str
    description: str= Field(description="description of the product")

class ProductsLists(BaseModel):
    products : list[ProductInfo]

class ProductAttribute(BaseModel):
  desire: str = Field(description="The desire product category")
  product_type: str = Field(description="Identify what type of product")
  list_variations: list[str] = Field(description="list of product variations")
    
def getProducts(info: ProductScope, verbose: Optional[bool] = False)->ProductsLists:
    llm = ChatOpenAI(temperature=0,model="gpt-4")
    # search = GoogleSerperAPIWrapper()
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS , memory=memory,verbose=verbose)
    parser = PydanticOutputParser(pydantic_object=ProductsLists)
    prompt = PromptTemplate(
        template="What are the top 5 for match the description :{product_scope} \n that matches the format. {format_instructions}\n\n",
        input_variables=["product_scope"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    _input = prompt.format_prompt(product_scope=f"{info.dict()}",format_instructions=parser.get_format_instructions())
    _output = agent.run(_input.to_string())
    # pass
    return parser.parse(_output)

def getAttribute(desire:str,verbose: Optional[bool] = False)->ProductAttribute:
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    search = SerpAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False)
    parser = PydanticOutputParser(pydantic_object=ProductAttribute)
    format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate(
    input_variables=["desire"],
    template = """You are a consumer with the following desire:{desire}. 
    Tasks:
    1. Identify what is the product category.
    2. List 20 possible variations types of product that most people would take into consideration.
    3..output as {format_instructions}""",
    partial_variables={'format_instructions':format_instructions})
    _input = prompt.format_prompt(desire=desire,format_instructions=format_instructions)
    _output = agent.run(_input.to_string())

    return parser.parse(_output)


def getSerpProducts(products:ProductsLists):
    search = SerpAPIWrapper()


    def _search(name):
        params = {
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": "google_shopping",
        "google_domain": "google.com",
        "q": name
        }

        search = GoogleSearch(params)
        return search.get_dict()
    
    results = []
    products_list = products.products
    # would be better to use a map function or dask

    # for product in products_list:
    #     name = product.name
    #     results.append(_search(name))
    
    # map function
    results = list(map(
        lambda product: _search(product.name), 
        products_list
        ))
    # get shopping_results from each results and extend it together
    results = list(map(
        lambda result: result.get("shopping_results"), 
        results
        )) # get shopping_results from each results
    results = [item for sublist in results for item in sublist] # flatten the list


    return results
   