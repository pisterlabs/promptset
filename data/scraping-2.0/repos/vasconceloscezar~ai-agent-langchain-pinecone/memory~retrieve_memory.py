import gzip
import json
from dotenv import load_dotenv

load_dotenv()
import os

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader

import pinecone

from testing_pinecone import init_pinecone


def compress_metadata(metadata):
    print("Compressing metadata")
    metadata_str = json.dumps(metadata)
    compressed_metadata = gzip.compress(metadata_str.encode("utf-8"))
    return compressed_metadata


def decompress_metadata(compressed_metadata):
    print("Decompressing metadata")
    decompressed_metadata_str = gzip.decompress(compressed_metadata).decode("utf-8")
    return json.loads(decompressed_metadata_str)


init_pinecone()

openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

loader = TextLoader("./memory/output.txt")
documents = loader.load()
print(f"Loaded {len(documents)} documents")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
print("Splitting documents")
docs = text_splitter.split_documents(documents)
print(f"Split {len(docs)} documents into {text_splitter} ")

embeddings = OpenAIEmbeddings()

print("Creating Pinecone index")
index_name = "gptest"
docsearch = Pinecone.from_documents(
    docs, embeddings, index_name=index_name, metadata_transform=compress_metadata
)

query = "Quais alterações no cálculo de ICMS foram feitas?"
retrieved_docs = docsearch.similarity_search(
    query, metadata_transform=decompress_metadata
)


template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".
Context: {context}
Question: {query}
Answer: """

prompt_template = PromptTemplate(
    input_variables=["context", "query"], template=template
)

responses = []
for doc in retrieved_docs:
    context = doc.page_content
    prompt = prompt_template.format(context=context, query=query)
    response = openai(prompt)
    responses.append(response)

print(responses)
