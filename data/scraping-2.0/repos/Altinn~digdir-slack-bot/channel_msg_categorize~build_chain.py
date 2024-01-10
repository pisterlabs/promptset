'''
===========================================
        Module: Chain functions
===========================================
'''
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from .prompts import categorize_new_message
from .llm import build_llm
from .config_chain import config

cfg = config()

def load_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=categorize_new_message,
                            input_variables=['question'])
    return prompt

def build_retrieval_choose_team(llm, prompt):
    dbqa = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return dbqa


def setup_dbqa():
    llm = build_llm()
    loaded_prompt = load_prompt()
    dbqa = build_retrieval_choose_team(llm, loaded_prompt)

    return dbqa


def query(dbqa, user_input):
    if cfg.MODEL_TYPE.startswith("gpt"):
        return dbqa(user_input)
    else:
        return dbqa({'query': user_input})
    
