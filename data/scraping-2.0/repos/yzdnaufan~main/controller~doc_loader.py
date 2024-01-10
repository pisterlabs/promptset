from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./doc/Lecture4.pdf")
pages = loader.load_and_split()