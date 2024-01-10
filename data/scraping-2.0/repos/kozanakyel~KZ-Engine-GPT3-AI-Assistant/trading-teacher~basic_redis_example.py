from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.redis import Redis

load_dotenv()

twitter_bearer_token = os.getenv('TW_BEARER_TOKEN')
openai_api_key = os.getenv('T_OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

loader = TextLoader("denemegpt.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

rds = Redis.from_documents(docs, embeddings, redis_url="redis://localhost:6379",  index_name='link04')
print(rds.index_name)

query = "What is the gptverse and their utilities?"
results = rds.similarity_search(query)
print(results[0].page_content.replace("\n", " "))