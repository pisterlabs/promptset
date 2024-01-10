from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from kanao.modules.get_api_key import get_api_key

def process_pdf(file_path):
    api_key = get_api_key()

    # Raise an error if API key is not provided
    if not api_key:
        raise ValueError('OpenAI API key is not provided in the configuration file.')

    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()

    # Create a ConversationalRetrievalChain with ChatOpenAI language model
    # and PDF search retriever
    pdf_search = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3),
        retriever=pdf_search.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
    )

    return chain

