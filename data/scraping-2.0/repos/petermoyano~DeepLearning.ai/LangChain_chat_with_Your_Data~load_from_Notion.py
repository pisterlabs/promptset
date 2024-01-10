from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("/path/to/your/notion/directory")
docs = loader.load()

print(docs[0].page_content[:500])
print(docs.metadata)
