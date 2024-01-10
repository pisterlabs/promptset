import os
import time
import config
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def create_base(file_path, kb_name):
    """
    创建PDF文件向量库
    :param kb_name:
    :param file_path: 文件路径
    :return:
    """
    try:
        print(f"file: {file_path}")
        print(
            "Start building vector database... %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        # loader = UnstructuredFileLoader(file_path, model="element")
        loader = PyPDFLoader(file_path)
        print("loader...")
        docs = loader.load()
        print(f"docs: {docs}")
        documents = []
        persist_directory = os.path.join(config.config["knowledge_path"], kb_name)

        # 向量化
        embeddings = HuggingFaceEmbeddings(model_name=config.config["text2vec"])

        if os.path.exists(persist_directory):
            print("找到了缓存的索引文件，加载中……")
            # return FAISS.load_local(persist_directory, embeddings)
            return {"status": 200, "message": "找到了缓存的索引文件，创建成功"}

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200
        )
        docs = text_splitter.split_documents(docs)
        documents.extend(docs)

        # 构造向量库+conversation_id
        vectordb = FAISS.from_documents(documents, embeddings)
        vectordb.save_local(persist_directory)
        print(
            "Vector database building finished. %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )

        return {"status": 200, "message": "知识库创建成功"}

    except Exception as e:
        print("errr........")
        return {"status": 500, "message": "知识库创建失败"}
