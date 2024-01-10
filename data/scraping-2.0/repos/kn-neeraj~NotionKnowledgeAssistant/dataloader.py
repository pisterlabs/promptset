from langchain.document_loaders import NotionDirectoryLoader

######## 1. LOADING THE DATA
## [Langchain reference](https://python.langchain.com/docs/integrations/document_loaders/notion.html)

def notion_data_loader(directory_path):
  loader = NotionDirectoryLoader(directory_path)
  raw_notion_docs = loader.load()
  return raw_notion_docs
