import openai
import numpy as np
import os
from dotenv import load_dotenv
import os
from collections import deque
from typing import Dict, List, Optional, Any
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

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)


path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print('path_to_root: ', path_to_root)

# Load variables from the .env file
load_dotenv(os.path.join(path_to_root, '.env'))

# Access the variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")
# Access the variables

# Set the environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

classifier_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=1)
classifier_article_schema = {
    "name": "binary_classifier_schema",
    "description": "Binary Classifier for is relevant question",
    "type": "object",
    "properties": {
      "isRelevant": {
        "type": "boolean"
      },
      "Reason": {
        "type": "string",
        "description": "Reason for your decision"
      }
    },
    "required": ["isRelevant","Reason"]
  }

classifierprompt = PromptTemplate(
    template = """Role:You are a Binary Classifier,your goal is to classify if the given Question is relevant to any following context:
    1. Questions relating to the port operations of PSA, including warehouses,berths, and other port facilities.
    2. Live querys on visualzation of current port operations, including the number of ships, capacity, and other metrics.
    3. What-if questions on the impact of disruptions on port operations

    Question:{question}\n\nFeedback:{feedback}\n

    TASK: Given youre Role, Classify if the question. Think through and give reasoning for your decision. Must Output boolean value for isDisruptionEvent.
    """,
    input_variables=["question","feedback"]

)

questionClassifier = create_structured_output_chain(output_schema=classifier_article_schema,llm = classifier_llm,prompt=classifierprompt)

classifier_question_schema = {
    "name": "binary_classifier_schema",
    "description": "Binary Classifier for is relevant question",
    "type": "object",
    "properties": {
      "isRelevant": {
        "type": "boolean"
      },
      "Reason": {
        "type": "string",
        "description": "Reason for your decision"
      }
    },
    "required": ["isRelevant","Reason"]
  }

visualizationClassificationPrompt = PromptTemplate(
    template = """Role:You are a Binary Classifier,your goal is to classify if the given Question is relevant to any following context:
    1. What-if questions on the impact of disruptions on port operations
    2. Requires re-optimization to get the best results using Gurobi


    EXCEPTION QUESTION: if the question specifically mentions the following, it is not relevant:
    1. current port operations which warehouse at maximum capacity --> Immediately reutrn FALSE for isRelevant

    Question:{question}\n\nFeedback:{feedback}\n

    TASK: Given youre Role, Classify if the question. Think through and give reasoning for your decision. Must Output boolean value for isDisruptionEvent.
    """,
    input_variables=["question","feedback"]
    
)
questionGurobiClassifier = create_structured_output_chain(output_schema=classifier_question_schema,llm = classifier_llm,prompt=visualizationClassificationPrompt)