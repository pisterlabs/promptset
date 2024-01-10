from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from mvi_langchain import MomentoVectorIndex
from momento import VectorIndexConfigurations, CredentialProvider
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
load_dotenv()

def insert():
    raw_documents = TextLoader('data/sample.txt').load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = MomentoVectorIndex(embedding_function=OpenAIEmbeddings(),
        configuration=VectorIndexConfigurations.Default.latest(),
        credential_provider=CredentialProvider.from_environment_variable("MOMENTO_AUTH_TOKEN"),
        index_name="sample")

    _ = db.add_documents(documents=documents, ids=[f"sotu-chunk-{i}" for i in range(len(documents))])
    print('finish insert')
    
insert()