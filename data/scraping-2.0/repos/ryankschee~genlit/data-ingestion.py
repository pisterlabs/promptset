import box
import yaml
import os
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings

# Load the configuration file
with open('config.yml', 'r', encoding='utf8') as f:
    config = box.Box(yaml.safe_load(f))

# Load the OpenAI API key
os.environ['OPENAI_API_KEY'] = config.openai.api.key
    
def run_ingestion():
    
    # Load the documents
    loader = DirectoryLoader(
        config.data.path, 
        glob='*.pdf', 
        loader_cls=PyPDFLoader
        )
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.data.chunk.size, 
        chunk_overlap=config.data.chunk.overlap
        )
    texts = text_splitter.split_documents(documents)
    num_elements = len(texts)
    print(num_elements)
    # print(texts[11])
    
    # Load the embeddings using OpenAI
    # embeddings = OpenAIEmbeddings()
    
    # Load the embeddings using HuggingFace
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config.data.embeddings,
        model_kwargs={'device': 'cpu'}
    )
    
    # Store the embeddings into FAISS
    vectorstore = FAISS.from_documents(
        documents=texts, 
        embedding=embeddings
    )
    vectorstore.save_local(config.data.faiss.path)
    
    # Load the persisted FAISS vectorstore from disk
    vectorstore = None
    vectorstore = FAISS.load_local(
        folder_path=config.data.faiss.path,
        embeddings=embeddings
    )
    
    # Store the embeddings into Chroma
    # vectorstore = Chroma.from_documents(
    #     documents=texts,
    #     embedding=embeddings,
    #     persist_directory=config.data.chroma.path
    # )
    # vectorstore.persist()
    
    # Load the persisted Chroma vectorstore from disk
    # vectorstore = None
    # vectorstore = Chroma(
    #     persist_directory=config.data.chroma.path,
    #     embedding_function=embeddings
    # )
    
    # DEBUG: Make an attempt to query the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    docs = retriever.get_relevant_documents('Who can purchase Manulife ReadyBuilder?')
    for doc in docs:
        print('--- DOC ---')
        print(doc)
     
    
if __name__ == '__main__':
    run_ingestion()