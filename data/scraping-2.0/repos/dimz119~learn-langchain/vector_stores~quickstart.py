from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# requires `pip install chromadb`
loader = CSVLoader(file_path='./fortune_500_2020.csv')
raw_documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
openai_embedding = OpenAIEmbeddings()
db = Chroma.from_documents(documents, openai_embedding, persist_directory="./fortune_500_db")

# save to disk
db.persist()

db_conn = Chroma(persist_directory="./fortune_500_db", embedding_function=openai_embedding)
query = "What is JPMorgan Revenue?"
docs = db_conn.similarity_search(query)
print(docs)
print(docs[0].page_content)

# # retriever
# db_conn = Chroma(persist_directory="./fortune_500_db", embedding_function=openai_embedding)
# retriever = db_conn.as_retriever()
# result = retriever.get_relevant_documents('walmart')
# print(result[0].page_content)
# """
# rank: 1
# company: Walmart
# no_of_employees: 2,200,000.00
# rank_change: None
# revenues: 523,964.00
# revenue_change: 0.02
# profits: 14,881.00
# profit_change: 1.23
# assets: 236,495.00
# market_value: 321,803.30
# """