from llama_index.tools import QueryEngineTool, ToolMetadata
import openai
from LLM_Index_Loader import load_index
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = os.environ["OPENAI_API_KEY"]

# model = "gpt-4"  # chunk limit 8192
# model = "gpt-3.5-turbo"  # chunk limit 4096
# model = "text-davinci-003"  # chunk limit 4097
model = "ada"
max_chunk_size = 1024

index_version = "v1"
data_path = "./data/index/" + index_version + "/"

# load index
index = load_index()  # load_index_from_storage(storage_context)

# Create query engine off of index
query_engine = index.as_query_engine(similarity_top_k=1)

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name='ArmyRegs', description='Army Regulations')
    )
]

query1 = "What are the time constraints for processing a flipl? Provide the name of the Army Regulation this info is found in."
query2 = "What are the specific codes that can be validly entered into GCSS Army that represent an equipment fault according to AR 750-1?"
query3 = "What are the DD Form 200 processing time segments and their associated time constraints?"

response = query_engine.query(query1)

for node in response.source_nodes:
    print(node.doc_id)
    print(node.score)
    print(node.source_text)

print(response)

response = query_engine.query(query2)
print(response)


response = query_engine.query(query3)
print(response)

