import langchain
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, YoutubeLoader
from langchain.indexes import VectorstoreIndexCreator


txt_loader = TextLoader("text.txt", encoding="utf8")

index = VectorstoreIndexCreator().from_loaders([txt_loader])
print('index:')
print(index)
