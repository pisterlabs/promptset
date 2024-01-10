from langchain.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import DeepLake
url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embed_model = TensorflowHubEmbeddings(model_url=url)

db = DeepLake(dataset_path='./deeplk', \
                       embedding=embed_model)

def load_doc(filename,metatag):
  from langchain.document_loaders import TextLoader
  from langchain.text_splitter import CharacterTextSplitter

  loader = TextLoader(filename)
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  docs = text_splitter.split_documents(documents)
  for d in docs:
    for key, value in metatag.items():
      d.metadata[key] = value
  return docs


alice_docs = load_doc('alice_in_wonderland.txt' ,{"author": "Lewis", "book": "Alice"})
stone_docs = load_doc('Sorcerer_stone.txt' ,{"author": "Rowling", "book": "Stone"})

lista_docs = db.add_documents(alice_docs)
lists_docs = db.add_documents(stone_docs)

query = "What was the Mad Hatter's riddle about raven and writing desks?"
docs = db.similarity_search(query)
print(docs[0].page_content)

db.similarity_search(
    "What color was the Sorcerer stone?",
    filter={"metadata": {"author": "Rowling"}},
)


