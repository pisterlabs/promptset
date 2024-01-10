"""
Basic setup to use langchain to interact with OpenAI's API and just get embeddings for text.
"""

import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # type: ignore
text = "There's so many benefits of hiring a data scientist. You should definitely hire me!"
doc_embeddings = embeddings.embed_documents([text])


print(doc_embeddings)
