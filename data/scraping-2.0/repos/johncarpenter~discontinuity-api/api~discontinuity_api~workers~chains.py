
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import logging

logger = logging.getLogger(__name__)

STANDARD_PROMPT = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
def retrieval_qa(index: VectorStore, query: str, embeddings = OpenAIEmbeddings(), prompt_template:str = STANDARD_PROMPT) -> str:
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=index.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    logger.info("Launching Retrieval QA Query")
    response = qa.run(query)
    logger.info("Completed Retrieval QA Query")
    return response
    