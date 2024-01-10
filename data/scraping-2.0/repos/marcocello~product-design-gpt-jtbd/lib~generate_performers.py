import os
import json 
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")

model = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
     
You are an advacend AI able to syntetic generate very detailed, precised and realistic users. All the fields required by the user should be very detailed and realistics.
     """),
    
    ("human", """

Create {number} {job_performers}. This is the output format:
     
{{
    "job_performers" {{
     [
        "Name": 
        "Profession":
        "Industry":
        "Experience_level":
        "Description":
     ]
     }}
}}

{additional_prompt}

     """)
])

functions = [
    {
    "name": "job_performers_generator",
    "description": "Generate job performers",
    "parameters": {
        "type": "object",
        "properties": {
            "job_performers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Name": {
                            "type": "string",
                            "description": "Name and surname of the job performer"
                        },
                        "Profession": {
                            "type": "string",
                            "description": "Profession of the job performer"
                        },
                        "Industry": {
                            "type": "string",
                            "description": "Industry of the job performer"
                        },
                        "Experience_level": {
                            "type": "string",
                            "description": "Experience level of the job performer"
                        },
                        "Description": {
                            "type": "string",
                            "description": "Full description of the job performer"
                        },
                    }
                }
            },
        },
        "required": ["job_performers"]
    },
    },
]

chain = (
    prompt 
    | model
)