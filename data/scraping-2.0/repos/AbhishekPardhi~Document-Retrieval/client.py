import os
import qdrant_client
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import VectorDBQA, OpenAI

# load enviorment variables

load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# create a qdrant client

client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# create collection

vectors_config = qdrant_client.http.models.VectorParams(
    size=1536,
    distance=qdrant_client.http.models.Distance.COSINE,
)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=vectors_config,
)

# create vector store

embeddings = OpenAIEmbeddings()

vector_store = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# add documents to vector store

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        cunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

file_path = os.getenv('FILE_PATH')

loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})

data = loader.load()

texts = get_chunks(data)

vector_store.add_texts(texts)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)


qa = VectorDBQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    vectorstore=vector_store,
    restur_source_documents=False,
)

question = 'what is the best product for hair growth?'
print('>', question)
print(qa.run(question), end="\n\n")
