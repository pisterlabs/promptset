from app.services.search import *
from app.config import *
from app.utils.utils import *

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

def get_llm(model_name=DEFAULT_MODEL_NAME,temperature=DEFAULT_MODEL_TEMPERATURE):
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
    )
    return llm

def get_chain(llm):
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    return chain

def retrieve_answers(documents, query):
    doc_search = retrieve_query(documents, query=query)
    llm = get_llm()
    chain = get_chain(llm=llm)
    response = chain.run(input_documents=doc_search, question=query)
    return response

def find_answer_to_query(query):
    docs = read_doc(DOCUMENTS_DIRECTORY_PATH)
    documents = chunk_data(docs=docs)
    answer = retrieve_answers(documents, query=query)
    return answer
