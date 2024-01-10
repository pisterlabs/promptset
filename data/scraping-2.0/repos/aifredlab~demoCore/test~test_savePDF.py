from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("opengpt/test.pdf")
documents = loader.load_and_split()
# from langchain.document_loaders import AmazonTextractPDFLoader
# loader = AmazonTextractPDFLoader("example_data/alejandro_rosalez_sample-small.jpeg")
# documents = loader.load()

total_characters = sum(len(content.page_content) for content in documents)
total_word = sum(len(content.page_content.split()) for content in documents)

print(f"total page : {len(documents)}")
print(f"total word : {total_word}")
print(f"total characters : {total_characters}")
print(f"price : ${total_characters / 1000 * 0.0001}") #$0.0001 

vector_db = Milvus.from_documents(
    documents,
    OpenAIEmbeddings(),
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

query = "의무보험이란?"
docs = vector_db.similarity_search(query)

print(f"요청 : {query}")
print(f"응답 : {docs[0].page_content}")
