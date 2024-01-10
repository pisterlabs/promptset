from langchain.document_loaders import HNLoader

loader = HNLoader("https://news.ycombinator.com/item?id=37321028")
data = loader.load()
print(data[0].page_content)
