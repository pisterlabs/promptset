from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.taobao.com//")
data = loader.load()
print(data)