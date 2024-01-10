from langchain.document_loaders import PDFMinerLoader
loader = PDFMinerLoader("doc_class.pdf")
data = loader.load()