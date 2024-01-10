"""
File that contains the tools to query for risk types.
"""
import logging

from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone 

from src.logic.config import secrets as config_secrets

class ToolSearchRiskTypes(BaseTool):
    name = "Search Risk Types"
    description = "useful for when you need information for a specific risk type"

    def run_find_type(self, query: str) -> str:
        """Queries a vectorstore to find an associated risk type. The tool takes an object of interest 
        (e.g. price increases, brand reputation, etc.) as input and returns the relevant risk type."""
        if not isinstance(query, str) or not type:
            raise ValueError("Argument type must be a non empty string")
        try:
            pinecone.init(
                api_key = config_secrets.read_pinecone_credentials(),
                environment = "us-west4-gcp"
            )
            index_name: str = "index-risk"
            namespace: str = "risk-types"
            embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
                openai_api_key = config_secrets.read_openai_credentials()
            )
            index: Pinecone = Pinecone.from_existing_index(index_name = index_name, embedding = embeddings, namespace = namespace)
            qa = RetrievalQA.from_chain_type(
                llm = ChatOpenAI(), 
                chain_type = "map_reduce", 
                retriever = index.as_retriever(),
                verbose = True
            )
            result: str = qa.run(f"What type of risk is associated with {query}. Please provide a detailed explanation.")
        except ValueError as e:
            logging.error(e)
            raise ValueError(f"Error: {e}") from e
        return result
    
    def _run(self, query: str) -> str:
        """Queries a vectorstore for detailed explanations of various risk types. These 
        descriptions aid in the risk assessment process. The tool takes a risk type (e.g. market risk
        as input and returns a detailed description of a relevant risk type."""
        if not isinstance(query, str) or not type:
            raise ValueError("Argument type must be a non empty string")
        try:
            pinecone.init(
                api_key = config_secrets.read_pinecone_credentials(),
                environment = "us-west4-gcp"
            )
            index_name: str = "index-risk"
            namespace: str = "risk-types"
            embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
                openai_api_key = config_secrets.read_openai_credentials()
            )
            index: Pinecone = Pinecone.from_existing_index(index_name = index_name, embedding = embeddings, namespace = namespace)
            qa = RetrievalQA.from_chain_type(
                llm = ChatOpenAI(), 
                chain_type = "map_reduce", 
                retriever = index.as_retriever(),
                verbose = True
            )
            result: str = qa.run(f"Please provide a detailed explanation of the risk type {query}.")
        except ValueError as e:
            logging.error(e)
            raise ValueError(f"Error: {e}") from e
        return result
    
    def _arun(self, query: str) -> None:
        raise NotImplementedError("This tool does not support asynchronous execution.")

    def get_mitigation_strategies(self, risk_type:str) -> str:
        """Queries a vectorstore for mitigation strategies for a specific risk type. The tool takes a risk type 
        (e.g. market risk) as input and returns a list of mitigation strategies."""
        if not isinstance(risk_type, str) or not type:
            raise ValueError("Argument risk_type must be a non empty string")
        try:
            pinecone.init(
                api_key = config_secrets.read_pinecone_credentials(),
                environment = "us-west4-gcp"
            )
            index_name: str = "index-risk"
            namespace: str = "risk-types"
            embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
                openai_api_key = config_secrets.read_openai_credentials()
            )
            index: Pinecone = Pinecone.from_existing_index(index_name = index_name, embedding = embeddings, namespace = namespace)
            qa = RetrievalQA.from_chain_type(
                llm = ChatOpenAI(), 
                chain_type = "map_reduce", 
                retriever = index.as_retriever(),
                verbose = True
            )
            result: str = qa.run(f"Please provide a detailed explanation of the mitigation strategies for risk type {risk_type}.")
        except ValueError as e:
            logging.error(e)
            raise ValueError(f"Error: {e}") from e
        return result