"""
This is a simple version of the customer support agent, which was built using `Agents` module.

This project was influenced by the following project: https://langchain-langchain.vercel.app/docs/use_cases/agents/sales_agent_with_context.html
"""
import os

import dotenv

dotenv.load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


from typing import Dict, List, Any

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import Cohere
from langchain.memory import ChatMessageHistory
