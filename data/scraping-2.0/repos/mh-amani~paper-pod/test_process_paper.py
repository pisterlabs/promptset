from langchain.document_loaders import PyPDFLoader

PDF_URL = "https://arxiv.org/pdf/1711.00937.pdf"

loader = PyPDFLoader(PDF_URL)
pages = loader.load_and_split()

print(pages)