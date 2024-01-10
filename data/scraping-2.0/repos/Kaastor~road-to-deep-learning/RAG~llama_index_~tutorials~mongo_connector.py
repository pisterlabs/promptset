import logging
import sys

from llama_index import SummaryIndex, ServiceContext
from llama_index.llms import OpenAI

import os
import openai

from mongo_client import SimpleMongoReader

'''
- Cannot load lists (only strings)
- If field is missing in record gives error
- must add a PromptTemplate to give information about underlying schema, explanation of fields
'''

openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# openai.log = "debug"

host = "mongodb://127.0.0.1"
port = 27017
db_name = "C03i2p9bk-drive"
collection_name = "file_meta"
# query_dict is passed into db.collection.find()
query_dict = {"owner": {"$eq": "susan@generalaudittool.com"}}
field_names = ["owner", "mimeType", "type", "title", "fileSize"]
reader = SimpleMongoReader(host, port)
documents = reader.load_data(
    db_name, collection_name, field_names, query_dict=query_dict
)

# define LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

index = SummaryIndex.from_documents(documents, service_context=service_context)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(similarity_top_k=5)
# response = query_engine.query("Count how many there is each mimeType used. Return information in format: "
#                               "`<mimeType>: <number_of_occurences>`")
# response = query_engine.query("What type of files susan has? Tell all of the types you can find."),
response = query_engine.query("What largest file susan has? "
                              "fileSize field has information about file size. It's type is a number. "
                              "When you see a number field treat it's values as numbers, "
                              "perform mathematical operations on them."
                              "Tell me it's size and title. ")
                              #"If you do not know the answer, just say `I do not know.`"),
print(response)
