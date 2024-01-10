import openai
import numpy as np
import os
from dotenv import load_dotenv
import os
from collections import deque
from typing import Dict, List, Optional, Any
from geopy.geocoders import Nominatim
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
# Langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain ,LLMCheckerChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    StringPromptTemplate
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from typing import Optional
from langchain.agents import AgentType, initialize_agent,AgentExecutor,BaseSingleActionAgent
from langchain.tools import tool
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union,Optional


from utils.Webscraper import Webscraper
from postgrest.exceptions import APIError


classifier_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=1)
classifier_article_schema = {
    "name": "binary_classifier_article_schema",
    "description": "Binary Classifier Schema for Article, is a disruption event or not",
    "type": "object",
    "properties": {
      "isDisruptionEvent": {
        "type": "boolean"
      },
      "disruptionType":{
        "type": "string",
        "description": "Type of disruption event. Must be one of the example the Full list given in conditions"
      },
      "Reason": {
        "type": "string",
        "description": "Reason for your decision"
      }
    },
    "required": ["isDisruptionEvent","disruptionType","Reason"]
  }

classifierprompt = PromptTemplate(
    template = """Role:You are a Binary Classifier,your goal is to classify if the given news article is a vaild disruption event article or not.
    Conditions:
    1. A disruption event can be defined as "An event that potentially impacts the supply chain and have a vaild disruption type".
    Full List of possible disruption types: a) Airport Disruption b) Bankruptcy c) Business Spin-off d) Business Sale e) Chemical Spill f) Corruption g) Company Split h) Cyber Attack i) FDA/EMA/OSHA Action j) Factory Fire k) Fine l) Geopolitical m) Leadership Transition n) Legal Action o) Merger & Acquisition p) Port Disruption q) Protest/Riot r) Supply Shortage a) Earthquake b) Extreme Weather c) Flood d) Hurricane e) Tornado f) Volcano g) Human Health h) Power Outage.
    2. The disruption event the news article is reporting, must be a 'live' event, meaning it is currently happening. Not an article reporting on a past event.

    Article Title:{articleTitle}\n{articleText}\nEnd of article\n\nFeedback:{feedback}\n

    TASK: Given youre Role and the Conditions, Classify if the given news article is a vaild disruption event article or not.A vaild disruption event article is classified as "An event that potentially impacts the supply chain and have a vaild disruption type",  Select the disruption type only based on the given full list of possible disruption types. Think through and give reasoning for your decision. Must Output boolean value for isDisruptionEvent.
    """,
    input_variables=["articleTitle","articleText","feedback"]
)
articleClassifier = create_structured_output_chain(output_schema=classifier_article_schema,llm = classifier_llm,prompt=classifierprompt)

location_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=1)
locationExtractorSchema = {
    "name": "locationExtractorSchema",
    "description": "Format and extract location of disruption from the given text",
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "Description": "Location of Disruption Event.Location should include any landmarks,cities, countries and addresses, output an address searchable in googleMaps be as specific as possible."
        }
    },
    "required": ["location"]
}
locationExtractorPrompt = PromptTemplate(
    template = """Role:You are a Location Extractor,your goal is to extract the location of the disruption event from the given text. Location of Disruption Event. Examples: 1.French Pass, New Zealand 2.Xiamen Fujian Chain,3.Perry, Florida, USA. \n\nArticle Title:{articleTitle}\n{articleText}\nEnd of article\n\nTask: Extract Location of disruption event.Location should include any landmarks,cities, countries and addresses, output an address searchable in googleMaps be as specific as possible.Feedback:{feedback}""",
    input_variables=["articleTitle","articleText","feedback"]
)

locationExtractor = create_structured_output_chain(output_schema=locationExtractorSchema,llm = location_llm,prompt=locationExtractorPrompt)


eventDetails_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=1)
eventDetails_schema = {
    "name": "eventDetails_schema",
    "description": "Format and extract the disruption event details from the given article",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "Description": "Name of Disruption Event, Includes some sort of key identifier"
        },
        "disruptionType": {
          "type": "string",
          "description": "Type of disruption event. Max 3 words"
        },
        "severity": {
          "type": "string",
          "description": "Quantifiable Severity metrics of the disruption event."
        },
        "radius": {
            "type": "number",
            "Description": "Estimated Radius of Disruption Event, in KM"
        }
    },# TODO: Get actual disruption Event Date, and accurate loop
    "required": ["name", "disruptionType","severity","radius"]
}

eventDetailsPrompt = PromptTemplate(
    template = """Role:You are a Disruption event News Analyst, your goal is to extract the details of the disruption event from the given article. \n\nArticle Title:{articleTitle}\n{articleText}\nEnd of article\n\nFeedback:{feedback}\n\nTask: Extract the details of the disruption event from the given article.Details include:\n1.Name of Disruption Event, Includes some sort of key identifier,with indication of severity\n2.Type of disruption event ,Max 3 words.\n3.Quantifiable Severity metrics of the disruption event.Extract multiple metrics,E.g Casualities,Cost damage etc.Example Severity:"Magnitude of 5.6, Depth of 170km. tremor was felt widely across parts, but no damage was caused overall"\n4.Estimated Radius of imapact of Disruption Event, in KM Must be greater then 0.Example: 100""",
    input_variables=["articleTitle","articleText","feedback"]
)

eventDetails = create_structured_output_chain(output_schema=eventDetails_schema,llm = eventDetails_llm,prompt=eventDetailsPrompt)