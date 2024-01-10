# 工具检索器(tool-retriever)
#   我们将使用向量存储来为每个工具描述创建嵌入。
#   然后，对于传入的查询，我们可以为该查询创建嵌入，并进行相关工具的相似性搜索。

import tools

# 设置环境
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(tools.ALL_TOOLS)]

# [
#     Document(page_content='useful for when you need to answer questions about current events', metadata={'index': 0}),
#     Document(page_content='a silly function that you can use to get more information about the number 0', metadata={'index': 1}),
#     Document(page_content='a silly function that you can use to get more information about the number 1', metadata={'index': 2}),
#     ......
# ]

vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vector_store.as_retriever()

# 测试搜索前十个和输入查询最相关的内容
top10SimilarityVal = vector_store.search(query="whats the weather?", search_type="similarity", k=10)
print("Top 10 Similarity Value ->", top10SimilarityVal)


# 根据传递的查询获取最相关的工具
def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [tools.ALL_TOOLS[d.metadata["index"]] for d in docs]


# 测试
if __name__ == '__main__':
    print(get_tools("whats the weather?"), "\n")

    # [
    #     Tool(name='Search', description='useful for when you need to answer questions about current events', args_schema=None, return_direct=False, verbose=False, callbacks=None, callback_manager=None, func=<bound method SerpAPIWrapper.run of SerpAPIWrapper(search_engine=<class 'serpapi.google_search.GoogleSearch'>, params={'engine': 'google', 'google_domain': 'google.com', 'gl': 'us', 'hl': 'en'}, serpapi_api_key='xxxx', aiosession=None)>, coroutine=None),
    #     Tool(name='foo-95', description='a silly function that you can use to get more information about the number 95', args_schema=None, return_direct=False, verbose=False, callbacks=None, callback_manager=None, func=<function fake_func at 0x104fd5da0>, coroutine=None),
    #     Tool(name='foo-12', description='a silly function that you can use to get more information about the number 12', args_schema=None, return_direct=False, verbose=False, callbacks=None, callback_manager=None, func=<function fake_func at 0x104fd5da0>, coroutine=None),
    #     Tool(name='foo-15', description='a silly function that you can use to get more information about the number 15', args_schema=None, return_direct=False, verbose=False, callbacks=None, callback_manager=None, func=<function fake_func at 0x104fd5da0>, coroutine=None)
    # ]

    print(get_tools("whats the number 13?"), "\n")
