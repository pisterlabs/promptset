import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import AI21
from utils import process_llm_response
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS
from langchain.embeddings import HuggingFaceEmbeddings


load_dotenv()
MAX_OP_TOKENS_FOR_AI21 = os.environ.get('AI_21_MAX_OUTPUT_TOKENS')
ai21_api_key = os.environ.get("ai21_api_key")
# llm = AI21(model="j2-jumbo-instruct", maxTokens=MAX_OP_TOKENS_FOR_AI21,ai21_api_key=ai21_api_key)
llm = AI21(model="j2-jumbo-instruct", ai21_api_key=ai21_api_key)
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')


def handleChat(modelPath, query):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    res = qa(query)
    process_llm_response(res)
    return res['result']
