import os
import openai
import pinecone

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("OPENAI_API_BASE")

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT 

INDEX_NAME = "hackaton-anthropic"
EMBEDDINGS_MODEL = "text-embedding-ada-002"
text_field = "text"

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(deployment_id=EMBEDDINGS_MODEL, 
                                chunk_size=1,
                                openai_api_key=API_KEY,
                                openai_api_base=RESOURCE_ENDPOINT
                                )

class ContentMatchingService:

  def get_content_related_to_question(self, question: str) -> str:
    vectorstore = Pinecone(index, embeddings.embed_query, text_field)

    docs = vectorstore.similarity_search(question, k=1)
    return docs[0].page_content
