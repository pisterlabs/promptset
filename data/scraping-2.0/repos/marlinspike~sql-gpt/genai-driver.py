import requests
from pydantic import BaseModel, Field
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import json
from dotenv import load_dotenv
load_dotenv()
from pydantic.json import pydantic_encoder
import sys
from rich import print_json
from rich.console import Console
from rich.syntax import Syntax
from typing import List, Optional

console = Console()

class Registration(BaseModel):
    license_plate: str = Field(...)
    registration_date: str = Field(...)
    expiry_date: str = Field(...)

class Vehicle(BaseModel):
    make: str = Field(...)
    model: str = Field(...)
    year: int = Field(...)
    registration: Registration = Field(...)

class Person(BaseModel):
    name: str = Field(...)
    address: str = Field(...)
    email: str = Field(...)
    alias: str = Field(...)
    vehicles: List[Vehicle] = Field(...)  # A list of Vehicle objects

    class Config:
        json_encoders = {
            # Your custom encoders if needed, for example:
            # datetime: lambda v: v.isoformat(),
        }

    def to_json(self):
        # Use model_dump() to get the model's data as a dict
        model_dict = self.model_dump(by_alias=True)
        # Use json.dumps() to convert the dict to a JSON string
        return json.dumps(model_dict)
        #return self.json()

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))

def parse_natural_language_to_json(natural_language_str: str) -> Optional[str]:
    chat = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0301")
    json_translate_str = """You are a JSON assistant. Convert this natural language description to JSON. 
    Here are the classes you should use. Ensure that the JSON you generate follows the schema of these classes:
    --
    class Registration(BaseModel):
    license_plate: str = Field(...)
    registration_date: str = Field(...)
    expiry_date: str = Field(...)

class Vehicle(BaseModel):
    make: str = Field(...)
    model: str = Field(...)
    year: int = Field(...)
    registration: Registration = Field(...)

class Person(BaseModel):
    name: str = Field(...)
    address: str = Field(...)
    email: str = Field(...)
    alias: str = Field(...)
    vehicles: List[Vehicle] = Field(...)  # A list of Vehicle objects
    --

    Description: {nl_input}"""

    json_translate_template = ChatPromptTemplate.from_template(json_translate_str)
    json_translate_messages = json_translate_template.format_messages(nl_input=natural_language_str)
    response = chat(json_translate_messages)
    return response.content

def communicate_with_legacy_api(json_payload: str, show_xml=True):
    headers = {'Content-Type': 'application/xml'}
    xml_payload = json_to_xml_with_llm(json_payload)
    if show_xml:
        print("XML Response Payload:")
        syntax = Syntax(xml_payload, "xml", theme="monokai", line_numbers=True)
        console.print(syntax)  # Optionally print the XML payload
    response = requests.post('http://127.0.0.1:5001/person', data=xml_payload, headers=headers)
    return xml_to_json_with_llm(response.content) if response.ok else None

def json_to_xml_with_llm(json_payload: str):
    chat = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo-0301")
    xml_translate_str = """You are an XML and JSON conversion bot. Convert this JSON to XML. Add some data (xml nodes) in there that looks like something a mainframe app would need. JSON: {json_input}"""

    xml_translate_template = ChatPromptTemplate.from_template(xml_translate_str)
    xml_translate_messages = xml_translate_template.format_messages(json_input=json_payload)
    response = chat(xml_translate_messages)
    return response.content

def xml_to_json_with_llm(xml_content: str):
    chat = ChatOpenAI(temperature=0.4)
    json_translate_str = """You are a JSON assistant. Convert this XML to JSON. XML: {xml_input}"""

    json_translate_template = ChatPromptTemplate.from_template(json_translate_str)
    json_translate_messages = json_translate_template.format_messages(xml_input=xml_content)
    response = chat(json_translate_messages)
    return response.content

def main():
    person = None
    show_xml = '--show-xml' in sys.argv  # Check if --show-xml is in the command line arguments
    
    natural_language_input = "\nJohn Doe lives at 123 Main St, his email is johndoe@example.com, his alias is JD. He has a 2020 Toyota Camry with license plate XYZ 1234, registered on 2020-01-01 and expiry on 2025-01-01."
    print(f"Natural Language Input: {natural_language_input}\n")
    user_input_json_str = parse_natural_language_to_json(natural_language_input)
    if user_input_json_str:
        try:
            person = Person.from_json(user_input_json_str)
        except ValueError as e:
            print(f"Error parsing JSON: {e}")
            return
    print("Converted JSON from Natural Language Input:")
    print_json(f"{user_input_json_str}\n")
    

    #Not used, since we're parsing natural language input to get this JSON
    """
    user_input_json = {
        "name": "John Doe",
        "address": "123 Main St",
        "email": "johndoe@example.com",
        "alias": "JD",
        "vehicles": [
            {
                "make": "Toyota",
                "model": "Camry",
                "year": 2020,
                "registration": {
                    "license_plate": "XYZ 1234",
                    "registration_date": "2020-01-01",
                    "expiry_date": "2025-01-01"
                }
            },
            # You can add more vehicles here
        ]
    }
    """

    #person = Person.from_json(json.dumps(user_input_json))
    response_json = communicate_with_legacy_api(person.to_json(), True)
    response_dict = json.loads(response_json)

    if response_json:
        print("Successfully communicated with the legacy API")
        #response_person = Person.from_json(response_json)
        print(response_dict['Response'])
    else:
        print("Failed to communicate with the legacy API")

if __name__ == "__main__":
    main()
