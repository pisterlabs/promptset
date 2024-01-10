"""
@Time    : 2023/12/26 16:00
@Author  : yangzq80@gmail.com
@File    : milvus.py
"""
from langchain.document_loaders import TextLoader,WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Milvus

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pymilvus import Milvus

def get_embeddingModel():
   
    
    model = HuggingFaceBgeEmbeddings(
         # model_name = "BAAI/gte-large-zh"
        model_name='/home/ubuntu/yzq/models/m3e-base',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': True}, # set True to compute cosine similarity
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    return model

def text_splitter(file_path):
    # loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    # loader = WebBaseLoader("http://llm1.yangzhiqiang.tech/output_1703337422.html")
    loader = TextLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def main():
    print('Hello, World!')
    all_splits = text_splitter('test/demo-controller.java')
    # for s in all_splits:
    #     print(f'{s}\n')
    # embeddings = OpenAIEmbeddings()
    embeddings = get_embeddingModel()

    # 创建集合
    # vector_db = Milvus.from_documents(
    #     all_splits,
    #     embeddings,
    #     collection_name="collection_1",
    #     connection_args={"host": "n3", "port": "19530"},
    # )

    vector_db = Milvus(
        embeddings,
        connection_args={"host": "n3", "port": "19530"},
        collection_name="collection_1",
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents("组件")
    print("use retriver:",docs[0])

    query = "可继承"
    docs = vector_db.similarity_search(query)
    print(docs[0].page_content)

if __name__ == '__main__':
    main()