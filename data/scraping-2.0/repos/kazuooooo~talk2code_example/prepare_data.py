import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from dotenv import load_dotenv

load_dotenv()

def load_docs(src_dir: str) -> list:
    """
    Load documents from a dir.

    Parameters:
    root_dir: Root directory to load documents from.
    """
    docs = []
    for dirpath, dirnames, filenames in os.walk((src_dir)):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
    return docs


def documents_to_chunks(documents: list) -> list:
    """
    Split documents into chunks.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    return chunks


def chunks_to_embeddings(chunks: list, data_set_name: str) -> list:
    """
    Embed chunks and upload to Deeplake
    """
    print("Embedding...")
    embeddings = OpenAIEmbeddings()
    DeepLake.from_documents(
        chunks, embeddings, dataset_path=f"hub://{os.getenv('ACTIVELOOP_ACCOUNT_NAME')}/{data_set_name}"
    )
    print("Done!")


# ファイルをDocumentとして読み込む
docs = load_docs(os.getenv('SRC_DIR'))

# ドキュメントをチャンクに分割
chunks = documents_to_chunks(docs)

# Embeddingして、DeepLakeにアップロード
embeddings = chunks_to_embeddings(chunks, os.getenv('DATA_SET_NAME'))
