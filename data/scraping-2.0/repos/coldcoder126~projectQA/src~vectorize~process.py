from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


def load_documents(dir="book"):
    # 加载文档
    loader = DirectoryLoader(dir)
    documents = loader.load()

    # 文本拆分
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=16)
    split_docs = text_spliter.split_documents(documents)
    print(split_docs[:2])
    return split_docs


def local_embedding():
    model_name = "thenlper/gte-large-zh"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf


def store_chroma(docs, embeddings, persist_directory="VectorDB"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


def load_chroma(persist_directory="VectorDB"):
    db = Chroma(persist_directory=persist_directory, embedding_function=local_embedding())
    return db


if __name__ == '__main__':
    documents = load_documents("../../static")
    embeddings = local_embedding()
    db = store_chroma(documents, embeddings)
