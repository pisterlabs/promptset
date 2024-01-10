from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdf_file.pdf")
pages = loader.load_and_split()
print(pages[0])