from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.schema.embeddings import Embeddings

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.schema.language_model import BaseLanguageModel

from langchain_qianwen import Qwen_v1


def persist_data_to_chroma(embeddings: Embeddings, persist_directory: str):
    loader = DirectoryLoader("./assets", glob="**/*.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"text length: {len(texts)}")
    print(f"data_count: {len(texts[0].page_content)}")

    # 使用 embedding engion 将 text 转换为向量
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    # 持久化向量 到本地目录
    db.persist()


def search_data_from_chroma(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    query: str,
):
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    rsp = qa.run({"query": query})
    print(rsp)


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-turbo",
    )

    persist_directory = "./vector_storage"
    embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
    )

    # 持久化数据到 chroma(开源向量存储引擎)
    # persist_data_to_chroma(embeddings, persist_directory)

    # 使用 llm 检索向量文本中的信息
    query = "文中工厂模式的例子有哪些??"
    search_data_from_chroma(llm, embeddings, query)
