import logging
import os
from datetime import datetime
from typing import Optional, Type

import pinecone
import streamlit as st
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.document import Document
from langchain.tools import BaseTool
from langchain.vectorstores import Pinecone
from pydantic import BaseModel
from xata.client import XataClient
import psycopg2
from langchain.llms import OpenAI

llm_model = st.secrets["llm_model"]
langchain_verbose = str(st.secrets["langchain_verbose"])
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["pinecone_environment"]
os.environ["PINECONE_INDEX"] = st.secrets["pinecone_index"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


class ReviewOutlineTool(BaseTool):
    name = "review_outline_tool"
    description = "Provide a well-structured outline for a review referring to original user query."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema
    
    def outline_chain(self):
        langchain_verbose = st.secrets["langchain_verbose"]
        openrouter_api_key = st.secrets["openrouter_api_key"]
        openrouter_api_base = st.secrets["openrouter_api_base"]

        selected_model = "anthropic/claude-2"
        # selected_model = "openai/gpt-3.5-turbo-16k"
        # selected_model = "openai/gpt-4-32k"
        # selected_model = "meta-llama/llama-2-70b-chat"

        llm_chat = ChatOpenAI(
            model_name=selected_model,
            temperature=0.7,
            streaming=True,
            verbose=langchain_verbose,
            openai_api_key=openrouter_api_key,
            openai_api_base=openrouter_api_base,
            headers={"HTTP-Referer": "http://localhost"},
            # callbacks=[],
        )
        prompt_template = """Provide a well-structured outline for a review based on original user query:
        "{text}",
        list a structured outline.
        """
        prompt =  PromptTemplate(input_variables=["text"], template=prompt_template)
        # Define LLM chain
        llm_chain = LLMChain(llm=llm_chat, prompt=prompt)
        return llm_chain


    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        
        outline_chain = self.outline_chain()
        response = outline_chain.run(query)
        return response

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
