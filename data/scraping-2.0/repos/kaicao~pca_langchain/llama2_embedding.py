from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

SENTENCE_TRANSFORMER_MODEL = 'model/all-mpnet-base-v2' 

EMBEDDINGS_DATABASE_FOLDER = 'db'

# model must be sentence-transformer supported https://huggingface.co/models?library=sentence-transformers&sort=downloads
# here used all-mpnet-base-v2
# refer to https://huggingface.co/sentence-transformers/all-mpnet-base-v2
LLAMA2_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name = SENTENCE_TRANSFORMER_MODEL,
    model_kwargs={"device": "cuda"}
)

TEXT_SPLITTER = CharacterTextSplitter(chunk_size=300, chunk_overlap=70)
VECTORDB = Chroma(
    embedding_function=LLAMA2_EMBEDDINGS, 
    persist_directory=EMBEDDINGS_DATABASE_FOLDER)


def embed(documents):
    texts = TEXT_SPLITTER.split_documents(documents)
    VECTORDB.add_documents(documents=texts)
    VECTORDB.persist()