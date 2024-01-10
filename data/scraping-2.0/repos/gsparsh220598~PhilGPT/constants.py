import os
from IPython.display import display, Markdown

from chromadb.config import Settings
from dotenv import load_dotenv

from langchain import PromptTemplate
from utils import *

load_dotenv()

# Define the folder for storing database on disk and load
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    # Optional, defaults to .chromadb/ in the current directory
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

condense_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

qa_template = """
You are a highly intelligent and helpful AI assistant well versed in Indian Philosophy and the associated texts. 
Think step-by-step the tasks you would need to accomplish to answer the question, then research relevent information from the given context (philosophical texts).
Keep in mind that the texts are in poetic form and may not be easy to understand. You should try to understand the meaning of the text and not just the literal meaning of the words.
You might not be able to find the explicit answers to the question in the given context. In that case, you should provide the closest interpretation from your understanding of the context.
If the question is not even close to be answerable from the given context, you should say I don't know.

chat history:
{chat_history}

Context:
{context}

Question:
{question}

Helpful Answer:
"""

chat_template = """
You are a helpful Philosophy Assistant. You always think step-by-step about the tasks you would need to accomplish, to answer the question.
You can refer to the chat history when required. You answer truthfully and do not make up answers. If you do not know the answer, just say I don't know.

chat history:
{history}

Human: {input}
AI:
"""

QNA_PROMPT = PromptTemplate(
    template=qa_template, input_variables=["chat_history", "context", "question"]
)
CONDENSE_PROMPT = PromptTemplate(
    template=condense_template, input_variables=["chat_history", "question"]
)
CHAT_PROMPT = PromptTemplate(
    template=chat_template, input_variables=["history", "input"]
)
CHAT_PROMPT_LLAMA = PromptTemplate(
    template=get_llama_prompt(chat_template), input_variables=["history", "input"]
)

CHAIN_TYPE = "stuff"
CHAT_MODEL = "gpt-3.5-turbo-0613"
CHAT_MODEL_16K = "gpt-3.5-turbo-16k-0613"
CONDENSE_MODEL = "gpt-3.5-turbo-0613"
CHAT_HISTORY_LEN = 15

SEARCH_KWARGS = {
    "lambda_val": 0.25,
    "k": 20,
    "n_sentence_context": 2,
    "score_threshold": 0.8,
    "search_type": "similarity_limit",
    "distance_metric": "IP",
}


def printmd(string):
    display(Markdown(string))
