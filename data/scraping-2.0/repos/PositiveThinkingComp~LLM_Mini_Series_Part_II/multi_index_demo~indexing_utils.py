from llama_index import ServiceContext, PromptHelper
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SimpleNodeParser
from config import AppConfig
import streamlit as st
from llama_index import ServiceContext, Document, PromptHelper
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index import  GPTVectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
from typing import List, Dict


class ServiceContextLoader:
    """
    This is a simple loader for the ServiceContext
    """
    def load(self) -> ServiceContext: 
        """
        This method loads the ServiceContext

        :return: Initialized ServiceContext for the Streamlit application
        """
        # Initialize an LLM Api Wrapper 
        llm = OpenAI(temperature=self.app_config.LLMConfig.temperature, 
                model=self.app_config.LLMConfig.model,
                max_tokens=self.app_config.LLMConfig.max_tokens) 

        # Initialize an LLM Embedding Model
        embed_model = OpenAIEmbedding()

        # Initialize a NodeParser from Documents to Nodes 
        node_parser = SimpleNodeParser.from_defaults(chunk_size=self.app_config.SimpleNodeParser.chunk_size, 
                                                    chunk_overlap=self.app_config.SimpleNodeParser.chunk_overlap)

        # Initialize a PromptHelper with the prompt parameters
        prompt_helper = PromptHelper(
            context_window=self.app_config.PromptHelper.context_window, 
            # num_output=256, 
            chunk_overlap_ratio=self.app_config.PromptHelper.chunk_overlap_ratio, 
            chunk_size_limit=self.app_config.PromptHelper.chunk_size_limit
        )

        # Initialize a ServiceContext for the query engine including the LLM, Embedding, NodeParser and PromptHelper
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser,
            prompt_helper=prompt_helper
        )
        return service_context

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config

@st.cache_resource
def create_multi_index(file_content_list: List[Dict], _service_context: ServiceContext,
                    top_k: int=3) -> SubQuestionQueryEngine:
    """
    This method creates a SubQuestionQueryEngine Multi-Index based on the indices for individual pdf pages

    :param app_config: AppConfig object configuring the Streamlit application
    :param file_content_list: List with the content per pdf file 

    :return: Multi-index query engine 
    """
    file_2_index = {}
    file_2_engine = {}
    for file_content in file_content_list:

        documents = [Document(text=file_content.get('text'))]

        index_name = file_content.get('index_name')
        engine_name = file_content.get('engine_name')
        title = file_content.get('title')

        # Initialize independently named GPTVectorStoreIndex objects on the fly 
        # e.g. index = GPTVectorStoreIndex.from_documents(documents)
        exec(f"{index_name} = GPTVectorStoreIndex.from_documents(documents)")

        # Initialize independently named query engines on the fly 
        # e.g. engine = index.as_query_engine(service_context=_service_context, similarity_top_k={top_k})
        exec(f"{engine_name} = {index_name}.as_query_engine(service_context=_service_context, similarity_top_k={top_k})")
        
        # Store each index and query engine in a dictionary 
        exec(f"file_2_index[title] = {index_name}")
        exec(f"file_2_engine[title] = {engine_name}")
    
    # Define a List of QueryEngineTools wrapping all individual pdf file indices 
    query_engine_tools = [
        QueryEngineTool(        
        query_engine=engine,
        metadata=ToolMetadata(
            name=title.replace(" ", "_"),
            description=title,
            ),
        )    
        for title, engine in file_2_engine.items()
    ]

    # Initialize a multi-index query engine based on all QueryEngineTools 
    s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

    return s_engine
