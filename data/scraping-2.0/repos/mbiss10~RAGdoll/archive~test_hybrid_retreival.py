# from langchain.llms import OpenAI
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from dotenv import load_dotenv
import pickle
import os
from langchain.retrievers import TFIDFRetriever
from langchain.schema import Document


load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = API_KEY


db = None
with open("./data/dev/db_cs_with_sources.pkl", "rb") as f:
    db = pickle.load(f)

query = "On what days does Deep Learning meet?"
res = db.similarity_search_with_relevance_scores(query, k=12)
print(res)

# retriever = db.as_retriever(search_kwargs={"k": 8})


# cs_data = None
# with open("./data/dev/cs_data.pkl", "rb") as f:
#     cs_data = pickle.load(f)

# texts, sources = list(zip(*cs_data))

# tfidf_retriever = TFIDFRetriever.from_texts(
#     texts, k=7)
# tfidf_retriever.docs = [Document(page_content=t, metadatas={"source": s})
#                         for t, s in zip(texts, sources)]


# query = "On what days does CSCI 381 meet?"
# docs = tfidf_retriever.get_relevant_documents(query)
# for doc in docs:
#     print(doc)


# for doc in docs:
#     print(doc.page_content)


# elasticsearch_url = "http://localhost:9200"
# retriever = ElasticSearchBM25Retriever.create(
#     elasticsearch_url, "langchain-index-4")

# retriever.add_texts(
#     texts, metadatas=[{"source": source} for source in sources])
# result = retriever.get_relevant_documents("foo")
