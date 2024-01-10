import os, pickle

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Set your OpenAI API key
def get_apikey():
    API_KEY = os.getenv('OPENAI_API_KEY') 
    return API_KEY

def get_embeddings(API_KEY):
    if not API_KEY:
        API_KEY = get_apikey()
        
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

    return embeddings

def get_vectorstore(name):
    store=None
    path = name + '.pkl'
    if os.path.exists(path):
        with open(path, "rb") as file:
            store = pickle.load(file)
    
    return store

def create_or_get_vectorstore(name, texts, embeddings, force=False):
    path = name + '.pkl'
    if os.path.exists(path) and force is not True:
        with open(path, "rb") as file:
            store = pickle.load(file)
    else:
        store = FAISS.from_texts([t.page_content for t in texts], embeddings)
    
        with open(path, "wb") as f:
            pickle.dump(store, f)

    return store


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

def get_prompt_context(background, safety):
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    
    template = """{}.You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say you're not sure. Don't try to make up an answer.{}."""
    template = template.format(background, safety)
    
    template = template + """
    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:"""
    
    QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

    return QA_PROMPT, CONDENSE_QUESTION_PROMPT

def chat(memory, retrievalchain):
    with get_openai_callback() as cb:
        while True:
            print("Human:")
            question = input()
            if question.lower() == "quit()":
                question = None
                break
            if question.lower() == "clear_history()":
                retrievalchain.memory.clear()
                question = None
                continue
            if question is not None and question != "" :
                print("AI:")
                print(retrievalchain.run(question))