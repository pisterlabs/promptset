from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from env import QDRANT_URL, OPENAI_API_KEY

client = QdrantClient(url=QDRANT_URL)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Qdrant(client=client, collection_name="insurance", embeddings=embeddings)

matching_docs = vector_store.similarity_search(query="what is Domiciliary Hospitalization?",k=5)
for doc in matching_docs: 
  print(doc.page_content)
  print("-"*20)