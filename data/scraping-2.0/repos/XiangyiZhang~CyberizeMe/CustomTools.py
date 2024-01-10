from typing import Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# get characteristics
class empty_input(BaseModel):
    pass

class get_characteristics(BaseTool):
    name = "get_characteristics"
    description = "useful for when you need to know about virtual character's characterisics"
    args_schema: Type[BaseModel] = empty_input

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("sync tool call is not implemented")
    
    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        with open('./config/characteristics.json') as file:
            data = json.load(file)
        return data
        
# get status
class get_status(BaseTool):
    name = "get_status"
    description = "useful for when you need to know about virtual character's current status, including mood, ongoing activity."
    args_schema: Type[BaseModel] = empty_input

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("sync tool call is not implemented")
    
    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        with open('./config/status.json') as file:
            data = json.load(file)
        return data

class input_name(BaseModel):
    name: str = Field()

# get relationship
class get_relationship(BaseTool):
    name = "get_relationship"
    description = "useful for when you need to know about a virtual character's relationship with someone."
    args_schema: Type[BaseModel] = input_name
    
    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("sync tool call is not implemented")
    
    async def _arun(self, name: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        with open('./config/relationship.json') as file:
            data = json.load(file)
        
        relationship = data.get(name.lower(), f"I don't know this person, {name}")
        return relationship

# get thoughts
class input_topic(BaseModel):
    topic: str = Field()

class get_thoughts(BaseTool):
    name = "get_thoughts"
    description = "useful for when you need to retrieve virtual character's past thoughts related to certain topic. "
    args_schema: Type[BaseModel] = input_topic

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("sync tool call is not implemented")
    
    async def _arun(self, topic: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        print(topic)
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="./chromadb/", embedding_function=embeddings)
        docs = db.similarity_search(topic, k=1)
        if not docs:
            return 'No related thoughts'
        print(docs[0].page_content)
        return (docs[0].page_content)
    