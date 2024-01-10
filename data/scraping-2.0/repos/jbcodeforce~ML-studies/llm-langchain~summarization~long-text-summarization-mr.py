import json
import os
import sys

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from bedrock.utils import bedrock, print_ww
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

'''
Split the long test into chunks
'''
shareholder_letter = "./data/2022-letter.txt"

with open(shareholder_letter, "r") as file:
    letter = file.read()
modelId = "anthropic.claude-v2"
llm = Bedrock(
    model_id=modelId,
    model_kwargs={
        'temperature': 0.3
    },
    client=boto3_bedrock,
)    


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
)

docs = text_splitter.create_documents([letter])
num_docs = len(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

print(
    f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
)

summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

print_ww(summary_chain.run(docs))

