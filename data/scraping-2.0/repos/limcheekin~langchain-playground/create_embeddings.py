from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from custom.embeddings.ctranslate2 import Ct2BertEmbeddings
from open.text.embeddings.openai import OpenAIEmbeddings
from open.text.embeddings.huggingface import E5Embeddings
import os
import pickle

PICKLE_FILE = os.environ['PICKLE_FILE']
EMBEDDINGS_MODEL_NAME = os.environ['EMBEDDINGS_MODEL_NAME']
E5_EMBED_INSTRUCTION = "passage: "
E5_QUERY_INSTRUCTION = "query: "
BGE_EN_EMBED_INSTRUCTION = ""
BGE_EN_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_ZH_EMBED_INSTRUCTION = ""
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def ingest_data():
    file_paths = None
    with open('html_files_index.txt', 'r') as file:
        file_paths = file.readlines()

    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=20,
        length_function=len
    )

    print("Load HTML files locally...")
    for i, file_path in enumerate(file_paths):
        file_path = file_path.rstrip("\n")
        doc = UnstructuredHTMLLoader(file_path).load()
        splits = text_splitter.split_documents(doc)
        docs.extend(splits)
        print(f"{i+1})Split {file_path} into {len(splits)} chunks")

    print("Load data to FAISS store")
    encode_kwargs = {"normalize_embeddings": True}
    if EMBEDDINGS_MODEL_NAME.startswith("universal-sentence-encoder"):
        print("Use universal-sentence-encoder model")
        embeddings = OpenAIEmbeddings(
            openai_api_base="http://localhost:8000/v1")
    elif "e5" in EMBEDDINGS_MODEL_NAME:
        print("Use E5Embeddings")
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDINGS_MODEL_NAME,
                                                   embed_instruction=E5_EMBED_INSTRUCTION,
                                                   query_instruction=E5_QUERY_INSTRUCTION,
                                                   encode_kwargs=encode_kwargs)
    elif EMBEDDINGS_MODEL_NAME.startswith("BAAI/bge-") and EMBEDDINGS_MODEL_NAME.endswith("-en"):
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDINGS_MODEL_NAME,
                                                   embed_instruction=BGE_EN_EMBED_INSTRUCTION,
                                                   query_instruction=BGE_EN_QUERY_INSTRUCTION,
                                                   encode_kwargs=encode_kwargs)
    elif EMBEDDINGS_MODEL_NAME.startswith("BAAI/bge-") and EMBEDDINGS_MODEL_NAME.endswith("-zh"):
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDINGS_MODEL_NAME,
                                                   embed_instruction=BGE_ZH_EMBED_INSTRUCTION,
                                                   query_instruction=BGE_ZH_QUERY_INSTRUCTION,
                                                   encode_kwargs=encode_kwargs)

    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL_NAME)

    # model_kwargs = {'device': 'cpu', 'compute_type': "int8"}
    # encode_kwargs = {'batch_size': 32,
    #                 'convert_to_numpy': True,
    #                 'normalize_embeddings': True}
    # embeddings = Ct2BertEmbeddings(
    #    model_name=EMBEDDINGS_MODEL_NAME,
    #    model_kwargs=model_kwargs,
    #    encode_kwargs=encode_kwargs
    # )
    store = FAISS.from_documents(docs, embeddings)

    print(f"Save to {PICKLE_FILE}")
    # store.save_local(PICKLE_FILE)
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(store, f)


if __name__ == "__main__":
    ingest_data()
