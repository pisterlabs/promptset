from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = "./.youtube"


def extract_video(urls):
    loader = GenericLoader(YoutubeAudioLoader(urls, SAVE_DIR),
                           OpenAIWhisperParser())
    docs = loader.load()

    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)
    return text


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def get_video_embeddings(text):
    # Split all text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_text(text)
    # Create a Vector Store
    docembeddings = FAISS.from_texts(splits, OpenAIEmbeddings())
    docembeddings.save_local("videos_faiss_index")
    docembeddings = FAISS.load_local("videos_faiss_index",
                                     OpenAIEmbeddings())
    return docembeddings
