from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
openai.api_key = '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'


docs = ["select user's quiz result", "select user's records of attendence", "select user's wrting records", "select user's reading records"]
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

db = FAISS.from_texts(docs, embeddings)

query = 'Check my attendence history'
docs = db.similarity_search(query)
print(docs[0].page_content)

db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
print(new_db.similarity_search('Check my attendence history')[0].page_content)