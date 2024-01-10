from langchain.embeddings import OpenAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

query_result = embeddings.embed_query("ITエンジニアについて30文字で教えて")

print(query_result)