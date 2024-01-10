from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

print(pages[1])