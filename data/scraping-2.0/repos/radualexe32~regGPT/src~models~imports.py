import sys
import re
import csv
import pandas as pd
import numpy as np
import os
import openai
import argparse
import json
import yaml
from getpass import getpass
from typing import Callable, List, Union, Any, Dict
from pydantic import BaseModel, Field, validator, root_validator
from langchain import PromptTemplate, SerpAPIWrapper, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredURLLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.agents import (
    AgentType,
    initialize_agent,
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
    create_json_agent,
)
from langchain.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
    BaseChatPromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import StructuredTool
from langchain.llms import OpenAI
from langchain.tools.requests.tool import RequestsGetTool, TextRequestsWrapper
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
from flask import Flask
from dotenv import load_dotenv
import gradio as gr
from gradio import themes
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
