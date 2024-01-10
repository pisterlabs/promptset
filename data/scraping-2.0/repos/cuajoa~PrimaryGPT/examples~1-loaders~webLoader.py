from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://clientes.primary.com.ar/escofondos/docs/api/consultiva/")

docs = loader.load()
print(docs[0].page_content[:500])