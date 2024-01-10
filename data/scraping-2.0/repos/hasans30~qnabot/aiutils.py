from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from decouple import config
import sys
import os

openai_api_key = config("OPENAI_API_KEY")


# it loads a directory of documents and return vector db
def load_documents():
    if not os.path.exists('data/'):
        print('data folder does not exist')
        return None
    loader = DirectoryLoader('data/', glob='**/*.txt', loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)  

def get_qa():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = load_documents()
    if texts is None:
        print('texts is none possibly due to data folder does not exist')
        return None
    docsearch = FAISS.from_documents(texts, embeddings)
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Create your Retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa

def query_my_question(queryText):
    qa=get_qa()
    if qa is None:
        print('qa is none possibly due to data folder does not exist')
        return 'unable to answer your question'
    query={"query": queryText}
    result=qa(query)
    return result['result']

# Compare this snippet from app.py:
if __name__ == '__main__':
    if len(sys.argv) <2 :
        print('not enough arguments')
        sys.exit(1)
    print(f'querying {sys.argv[1]}')
    print(query_my_question(sys.argv[1]))