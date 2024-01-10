# importing libraries
import os
import openai
from dotenv import load_dotenv
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, GPTTreeIndex

from gpt_index.readers.database import DatabaseReader
# use cases to gpt index data structures : GPTSimpleVectorIndex, GPTListIndex, GPTTreeIndex

# initializing variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
directory = 'data'

# general usage pattern of gpt index: this builds an index over the documents in the data folder
documents = SimpleDirectoryReader(directory).load_data() # step 1 load in documents
#index = GPTTreeIndex(documents)
index = GPTSimpleVectorIndex(documents) # step 2 index Construction - step 3 building indices on top of other indices (optional step)
#response = index.query("Please summarize the document", mode="summarize")
response = index.query("Please summarize how LaMDA works") # step 4 query the index
print(response)

# save to disk
index.save_to_disk('index_lambda.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index_lambda.json')