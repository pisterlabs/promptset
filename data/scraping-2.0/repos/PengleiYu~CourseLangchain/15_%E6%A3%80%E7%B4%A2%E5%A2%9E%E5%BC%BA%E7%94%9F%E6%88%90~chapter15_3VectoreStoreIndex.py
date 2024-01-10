# 矢量存储检索器：向量存储类的轻量级包装
# 第二章RAG的简化版
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader("../02_文档QA系统/docs/花语大全.txt", encoding='utf8', )
# 内部默认配置了相关参数：模型、文本分割器等
index = VectorstoreIndexCreator().from_loaders(loaders=[loader])

query = '玫瑰花的花语是什么？'
result = index.query(question=query)
print(result)

query = '茉莉花的花语是什么？'
result = index.query(question=query)
print(result)
