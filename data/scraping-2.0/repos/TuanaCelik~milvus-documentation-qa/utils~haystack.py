import streamlit as st

from haystack import Pipeline
from haystack.nodes import EmbeddingRetriever, PromptNode, PromptTemplate, AnswerParser
from milvus_haystack import MilvusDocumentStore
from utils.config import OPENAI_API_KEY

@st.cache_resource(show_spinner=False)
def start_haystack():
    document_store = MilvusDocumentStore()

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    template = PromptTemplate(prompt="deepset/question-answering", output_parser=AnswerParser())
    prompt_node = PromptNode(model_name_or_path="gpt-4", default_prompt_template=template, api_key=OPENAI_API_KEY, max_length=500)

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
    return query_pipeline

pipe = start_haystack()

@st.cache_data(show_spinner=True)
def query(question):
    params = {"Retriever": {"top_k": 5}}
    results = pipe.run(question, params=params)
    return results['answers']