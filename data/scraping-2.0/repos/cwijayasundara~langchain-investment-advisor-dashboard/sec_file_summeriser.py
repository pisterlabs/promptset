import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

doc_url = 'https://abc.xyz/assets/a7/5b/9e5ae0364b12b4c883f3cf748226/goog-exhibit-99-1-q1-2023-19.pdf'

llm = ChatOpenAI(temperature=0.0)


# run the summarizer chain
def summerise_large_pdf(fileUrl):
    loader = PyPDFLoader(fileUrl)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(texts)


with get_openai_callback() as cb:
    response = summerise_large_pdf(doc_url)
    print(response)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
