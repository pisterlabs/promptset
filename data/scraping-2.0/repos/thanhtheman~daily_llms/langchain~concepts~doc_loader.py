from langchain.document_loaders import HNLoader

loader = HNLoader("https://news.ycombinator.com/item?id=36645575")
data = loader.load()
print(data[2])
print(f"/n{data[0].metadata}")