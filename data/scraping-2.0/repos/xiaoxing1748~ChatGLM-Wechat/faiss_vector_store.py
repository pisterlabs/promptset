# https://python.langchain.com/docs/integrations/vectorstores/faiss
import document_loader
from langchain.vectorstores import FAISS
import embeddings


# 加载embedding
# embedding = embeddings.load()


# FAISS向量存储
def index(document_path, embedding=None):
    """return vector_store"""
    # 加载并分割文档
    text = document_loader.load_and_split_unstructured(document_path)
    # 输出分割完的文档
    print(text)
    # 构建向量存储
    if embedding is None:
        embedding = embeddings.load()
    vector_store = FAISS.from_documents(text, embedding)
    # 保存索引
    vector_store.save_local("faiss_index")
    return vector_store


# 加载索引
def load(embedding=None):
    """return vector_store"""
    if embedding is None:
        embedding = embeddings.load()
    return FAISS.load_local("faiss_index", embedding)


# 查询
def search(query, document_path, vector_store=None):
    """return docs"""
    if vector_store is None:
        vector_store = index(document_path)
    # 查询向量
    # docs = vector_store.similarity_search(query)
    # 查询带分数的向量
    docs = vector_store.similarity_search_with_score(query)
    return docs


# 加载索引并搜索
def load_and_search(query):
    """return docs"""
    docs = load().similarity_search_with_score(query)
    return docs


if __name__ == '__main__':
    # print(index("./document/news.txt").as_retriever())
    docs = ""
    # docs = search("摘要", "./document/news.txt")
    # docs = load_and_search("星期四")
    if docs:
        for doc in docs:
            print(doc)
        print("--------------------")
        context = []
        # 遍历docs中的每个元素，提取page_content并添加到context
        for doc in docs:
            context.append(doc[0].page_content)
        print(context)
