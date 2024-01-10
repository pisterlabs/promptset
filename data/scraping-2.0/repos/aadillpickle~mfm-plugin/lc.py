import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_ENV  
)

embeddings = OpenAIEmbeddings()
index_name = "mfm-embddings"

docsearch = Pinecone.from_existing_index(index_name, embeddings)

def get_relevant_info_from_question(query):
    relevant_info = []
    docs = docsearch.similarity_search_with_score(query, k=5)

    for doc in docs:
        print(doc[1])
        relevant_info.append({
                "snippet": doc[0].page_content,
                "podcast_title": doc[0].metadata["title"],
                "link": doc[0].metadata["description"] 
            })
    return relevant_info