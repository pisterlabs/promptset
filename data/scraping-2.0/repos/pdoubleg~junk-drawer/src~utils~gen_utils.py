import re
from io import BytesIO
from typing import List
import os
import spacy
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.embeddings import LangchainEmbedding
from langchain.docstore.document import Document


def get_sql_index_tool(sql_index, table_context_dict):
    table_context_str = "\n".join(table_context_dict.values())

    def run_sql_index_query(query_text):
        try:
            response = sql_index.as_query_engine(synthesize_response=False).query(query_text)
        except Exception as e:
            return f"Error running SQL {e}.\nNot able to retrieve answer."
        text = str(response)
        sql = response.extra_info["sql_query"]
        return f"Here are the details on the SQL table: {table_context_str}\nSQL Query Used: {sql}\nSQL Result: {text}\n"

    return run_sql_index_query


def get_llm(model_temperature):
    from langchain.chat_models import ChatOpenAI
    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(temperature=model_temperature, model_name="gpt-3.5-turbo-0613")

def get_llama_embeddings_model():
    return LangchainEmbedding(OpenAIEmbeddings())

def get_lc_embeddings_model():
    return OpenAIEmbeddings()


def wrap_text_in_html(text: List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


def find_eos_spacy(text):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    return [sent.end_char for sent in doc.sents]


def create_doc_from_pages(page_list):
    text = ''.join(page_list)
    doc = Document(text=text)
    return doc

