import os
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


os.environ["OPENAI_API_KEY"] = ""

def creating_model(docs):
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents=docs)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriver = db.as_retriever(search_type='mmr', search_kwargs={"k":2})

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type = "refine",
        retriever= retriver,
        return_source_documents=False
    )
    
    return qa

def question_answering(model, query):
    result = model({"query": query})
    
    return result

def load_document(path, name):
    try:
        filepath = os.path.join(path, name)
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        print("Document Read from {}".format(path))
        
        return documents
    except:
        print("Unable to Read Document")
