import os
from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

KEY = "your_key"
os.environ["OPENAI_API_KEY"] = KEY
VERBOSE=False

if __name__ == "__main__":

    # load the PDF and print out page content sample
    pdf_path = "./archive.pdf"
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    if VERBOSE:
        print(pages[0].page_content)

    # create embeddings and persist them 
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory="./vectordb")
    vectordb.persist()

    # quering the db via the model
    pdf_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9),vectordb.as_retriever())

    query = "What is the VideoTaskformer?"
    result = pdf_qa({"question": query, "chat_history": ""})
    print("Answer:")
    print(result["answer"])
