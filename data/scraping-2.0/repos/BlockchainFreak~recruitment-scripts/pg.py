import os
from typing import List, Tuple
import dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
dotenv.load_dotenv()


embeddings = OpenAIEmbeddings()
CONNECTION_STRING = os.environ["DATABASE_URL"]

# Initialize the vectorstore, create tables and store embeddings and
# metadata.

db = PGVector(
    connection_string=CONNECTION_STRING, 
    embedding_function=embeddings,
    collection_name="resumes-test1",
)

db.from_documents

db.add
print(dir(db))

exit(0)
# Create the index using HNSW. This step is optional. By default the
# vectorstore uses exact search.
db.create_hnsw_index(max_elements=10000, dims=1536, m=8, ef_construction =16, ef_search=16)

# Execute the similarity search and return documents
query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

print('query done')

print("Results:")
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)