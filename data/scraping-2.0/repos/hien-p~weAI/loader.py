from langchain.document_loaders import DirectoryLoader
# directory path
directory = 'data'


def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


documents = load_docs(directory)
print(len(documents))