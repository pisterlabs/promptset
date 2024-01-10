from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

## text splitter imports
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import  RetrievalQA
from langchain import OpenAI

## parsing csv to string
import pandas as pd



## laoding texts from local directory

def load_doc_txt():
    loader = DirectoryLoader('./assets', glob="**/*.txt", show_progress=True)
    docs = loader.load() 
    return docs

def load_plain_text(plain_text):
    doc = Document(page_content=plain_text)
    return [doc]

## create vector store for the documents and create embedding

def create_embedding(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    return db
    
## load csv file as query

def load_csv():
    loader = CSVLoader('./data_uploaded', glob="**/*.csv", show_progress=True, encoding="utf-8")
    loadedcsv = loader.load()
    return loadedcsv
    print (loadedcsv)


## converting csv to string

def load_csv_as_query(csv_file):
    df = pd.read_csv(csv_file)
    query = df.to_string(index=False)
    return query



## main function 
def main():
    # load documents
    docs = load_doc_txt()
    # print(docs)
    # create embedding
    docsearch = create_embedding(docs)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    query = "What is the contents of the file?"
    ans = qa.run(query)
    print(ans)
    
    
if __name__ == "__main__":
    main()
    
    

