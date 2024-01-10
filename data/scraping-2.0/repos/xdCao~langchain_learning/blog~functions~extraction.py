import configparser
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser, JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate

config = configparser.ConfigParser()
config.read('../../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')


class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")


class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")


funcs = [convert_pydantic_to_openai_function(Information)]


prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

llm = ChatOpenAI(openai_api_key=openai_api_key)
llm = llm.bind(functions=funcs, function_call={"name": "Information"})

chain = prompt | llm | JsonKeyOutputFunctionsParser(key_name="people")

response = chain.invoke({"input": "小明今年15岁，他的妈妈是张丽丽"})
print(response)
