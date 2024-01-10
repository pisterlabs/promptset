from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

extracted_data = load_pdf("data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

"""
#initializing the Pinecone
#run this block of code if do not have stored index in PINECONE 

pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name = "med-chatbot"

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks],
                                embeddings,
                                index_name=index_name)
                                
"""

