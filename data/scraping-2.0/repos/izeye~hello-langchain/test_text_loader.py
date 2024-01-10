from langchain.document_loaders import TextLoader

loader = TextLoader("./README.md")
output = loader.load()
print(output)
