import os

from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "dataset")

print(CUR_DIR, DATA_DIR)

CONSULT_DIR = os.path.join(DATA_DIR, "consult")
DRUG_DIR = os.path.join(DATA_DIR, "drug")

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "doctor-khu"

def upload_embedding_from_dir(file_path):
    loader = CSVLoader(file_path, encoding="utf-8")

    if loader is None:
        raise ValueError("Invalid file path")
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(), # 비용 발생하므로 허깅페이스 임베딩을 사용해도 될 듯
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # 현재 csv 파일만 이용하므로 csv 파일에 대해서만 진행
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                try:
                    upload_embedding_from_dir(file_path)
                    print(f"Uploaded {file_path}")
                except Exception as e:
                    print(f"Failed to upload {file_path} : {e}")
                    failed_upload_files.append(file_path)

if __name__ == "__main__":
    upload_embeddings_from_dir(DRUG_DIR)
    upload_embeddings_from_dir(CONSULT_DIR)