import re
import os
import json
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union

from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from openai.error import AuthenticationError
from langchain.docstore.document import Document
from pypdf import PdfReader
from sqlalchemy.engine.url import URL
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks.base import BaseCallbackManager
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

try:
    from .prompts import (
        COMBINE_QUESTION_PROMPT,
        COMBINE_PROMPT,
        COMBINE_CHAT_PROMPT,
        CSV_PROMPT_PREFIX,
        CSV_PROMPT_SUFFIX,
        MSSQL_PROMPT,
        MSSQL_AGENT_PREFIX,
        MSSQL_AGENT_FORMAT_INSTRUCTIONS,
        CHATGPT_PROMPT,
        BING_PROMPT_PREFIX,
        DOCSEARCH_PROMPT_PREFIX,
    )
except Exception as e:
    print(e)
    from prompts import (
        COMBINE_QUESTION_PROMPT,
        COMBINE_PROMPT,
        COMBINE_CHAT_PROMPT,
        CSV_PROMPT_PREFIX,
        CSV_PROMPT_SUFFIX,
        MSSQL_PROMPT,
        MSSQL_AGENT_PREFIX,
        MSSQL_AGENT_FORMAT_INSTRUCTIONS,
        CHATGPT_PROMPT,
        BING_PROMPT_PREFIX,
        DOCSEARCH_PROMPT_PREFIX,
    )

class SimulatorTool(BaseTool):
    """Tool for a HQ Simulator"""

    name = "@simulator"
    description = "useful when the questions includes the term: @simulator.\n"

    llm: AzureChatOpenAI

    def _run(self, query: str) -> str:
        try:
            agent_executor = create_python_agent(
                llm=OpenAI(temperature=0, max_tokens=1000),
                tool=PythonREPLTool(),
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            response = agent_executor.run(query)

            return response
        except Exception as e:
            response = str(e)

        return response

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ChatGPTTool does not support async")
