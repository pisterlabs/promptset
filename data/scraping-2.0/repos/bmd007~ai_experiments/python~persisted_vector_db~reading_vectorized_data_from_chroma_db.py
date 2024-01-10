from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

persist_directory = '../../local_data/vector_db_persistence'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
print(vector_db.get())
docs = vector_db.similarity_search("enrollment")
print('\n')
print('\n')
print('\n')
print('\n')
print('\n')
print(docs[0].page_content)
