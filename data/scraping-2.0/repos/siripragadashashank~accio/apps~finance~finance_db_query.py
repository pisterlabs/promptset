import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


client = chromadb.HttpClient(host='localhost', port=8000)


db = Chroma(client=client,
            collection_name="accio_financial_llama",
            embedding_function=embedding_function)

query = "what are the cash flow issues in Q3 of fy22"
docs = db.similarity_search(query)
print(docs[0].page_content)


