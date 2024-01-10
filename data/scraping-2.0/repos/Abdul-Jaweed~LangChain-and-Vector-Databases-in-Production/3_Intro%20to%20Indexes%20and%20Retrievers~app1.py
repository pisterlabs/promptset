# Streamlined Data Ingestion: Text, PyPDF,  Selenium URL Loaders, and Google Drive Sync


from langchain.document_loaders import TextLoader

loader = TextLoader("my_file.txt")

documents = loader.load()

