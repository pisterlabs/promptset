import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, validator, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.document_loaders import UnstructuredXMLLoader
import xml.etree.ElementTree as ET

# load dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo-16k-0613',temperature=0)

base_prompt = "You have to strictly respond with the appropriate url where the user should be taken from base url to fulfill their query, this is a non-negotiable. For example: http://www.logitech.com/en-in/products/keyboards.html"

def load_xml_file_as_string(file_path):
    try:
        with open(file_path, 'r') as file:
            xml_string = file.read()
            return xml_string
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# get sitemap string from xml:
xml_path = "../data/logi-sitemap.xml"
xml_string = load_xml_file_as_string(xml_path)
tree = ET.ElementTree(ET.fromstring(xml_string))

string_list = []
for item in tree.iter('url'):
    string_list.append(item.text)

single_xml = ''.join(string_list)

class WebNavigSchema(BaseModel):
    query: str = Field(description="base user query by the user")

class WebNavigator(BaseTool):
    name = "Web_navigator"
    description = "this will output a single line which will contain the url of the target page"
    args_schema: Type[WebNavigSchema] = WebNavigSchema

    def _run(
        self, 
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print("### web navigator tool ###")
        return llm(query + base_prompt + single_xml)

    async def _arun(
        self, 
        query: str,
        # run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")




