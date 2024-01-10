from gpt_index import GPTKeywordTableIndex, SimpleDirectoryReader
from IPython.display import Markdown, display
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader('data').load_data()
# load from disk
index = GPTKeywordTableIndex.load_from_disk('index.json')

while True:
    query = input("What would you like to ask DevX bot? ")
    response = index.query(query, response_mode="default", verbose=False)
    print("\n\nDevX Bot says:\n\n" + response.response + "\n\n\n")
