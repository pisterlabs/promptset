# Neo4J
#
# see https://python.langchain.com/docs/integrations/vectorstores/neo4jvector
#
import os
import getpass

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Neo4jVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

loader = TextLoader("state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# Neo4jVector requires the Neo4j database credentials

url = "<YOUR_URL>"
username = "neo4j"
password = "<YOUR_PASSWORD>"

# The Neo4jVector Module will connect to Neo4j and create a vector index if needed.
db = Neo4jVector.from_documents(
    docs, OpenAIEmbeddings(), url=url, username=username, password=password
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
