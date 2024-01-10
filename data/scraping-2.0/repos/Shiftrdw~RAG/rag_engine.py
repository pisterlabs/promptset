from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vectordb')
OLLAMA_PATH="http://localhost:11434"

def load_llm(llm_name = "llama2", **kwargs):
    
    match llm_name:
        case "llama2":
            return Ollama(
                base_url=OLLAMA_PATH,
                model=llm_name, 
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            )
        case "openai":
            return OpenAIChat(openai_api_key=kwargs["openai_api_key"])
        case _:
            raise Exception("LLM not found")
        
def load_hf_embeddings(model_name = None):
    model_name = "sentence-transformers/all-mpnet-base-v2" if model_name == None else model_name
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts, embedding_model=None):
    embedding = OpenAIEmbeddings() if embedding_model == None else embedding_model
    try:
        vectordb = Chroma.from_documents(
            texts, 
            embedding=embedding, 
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
    except Exception as e:
        Chroma().delete_collection()
        vectordb = Chroma.from_documents(
            texts, 
            embedding=embedding, 
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def query_llm(retriever, llm, messages, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': messages})
    result = result['answer']
    messages.append((query, result))
    return result
    