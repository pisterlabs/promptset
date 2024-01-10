import box
import yaml
import os
import openai
import instructor
from pydantic import BaseModel, Field
import pprint
from utils.openai_utils import json_gpt

from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from docs_qa.prompts import qa_template
from docs_qa.llm import build_llm
from utils.openai_utils import json_gpt


# module init
with open('docs_qa/config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

instructor.patch()
openai.api_key = os.environ['OPENAI_API_KEY_ALTINN3_DEV']
openai.api_base = os.environ['OPENAI_API_URL_ALTINN3_DEV']

pp = pprint.PrettyPrinter(indent=2)



async def generate_hypothetical_answer(user_input) -> str:    
    HA_INPUT = f"""Generate a hypothetical answer to the user's question. This answer will be used to rank search results. 
Pretend you have all the information you need to answer, but don't use any actual facts. Instead, use placeholders
like NAME did something, or NAME said something at PLACE. 

User question: {user_input}

Format: {{"hypotheticalAnswer": "hypothetical answer text"}}
"""

    hypothetical_answer = json_gpt(HA_INPUT)["hypotheticalAnswer"]

    return hypothetical_answer


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt},      
                                       verbose=False                                 
                                       )
    return dbqa

def setup_dbqa():
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa
