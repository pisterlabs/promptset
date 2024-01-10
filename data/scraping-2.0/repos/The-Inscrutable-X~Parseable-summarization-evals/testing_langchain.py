import os
import openai
import requests
import nltk
import json
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain

# nltk.download('punkt')

# Setup environment variables.
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
openai.api_key = key

# Use preexisting chain
llm = OpenAI(temperature=0)
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
from langchain.chains import AnalyzeDocumentChain
summarize_doc_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
with open("test_paper.txt") as f:
    paper = f.read()
out = summarize_doc_chain.run(paper)
print(out)
# Create loader for documents

# Create splitter for documents

# Split documents

# Import storage class and embedder. 

# Embed and store document splits.

# Create a prompt, import the prompt from the Hub