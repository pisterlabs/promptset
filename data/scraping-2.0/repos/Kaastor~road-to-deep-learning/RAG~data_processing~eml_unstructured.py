from langchain.document_loaders import UnstructuredEmailLoader


loader = UnstructuredEmailLoader("/home/przemek/GAT/emails/18a2cebb6c23bc7d/1692970714000.eml", mode="elements")
data = loader.load()
print()