import sys
import argparse

from dotenv import load_dotenv

import chromadb

from chromadb.config import Settings

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Parse Arguments

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description="Loads and/or queries data into/from ChromaDB using OpenAI",
    allow_abbrev=True
)

parser.add_argument("-l", "--load", help="load file", type=str, action="append")
parser.add_argument("-q", "--query", help="ask question", type=str, action="append")
parser.add_argument("-v", "--verbose", help="enable verbose mode", action="store_true")
args = parser.parse_args()

if (not args.load or len(args.load) == 0) and (not args.query or len(args.query) == 0):
    print(f"no file(s) to load and no queries specified")
    parser.print_usage()
    exit(1)


def print_verbose(message):
    if args.verbose:
        print(message)

# Load Configuration

load_dotenv()

# Setup Database Access

chroma_settings = Settings(chroma_api_impl="rest", chroma_server_host="localhost", chroma_server_http_port="8000")
chroma_client = chromadb.Client(chroma_settings)

# Load New Data

documents_to_load = []
if args.load:
    for fn in args.load:
        print_verbose(f"reading({fn})")
        loader = TextLoader(fn)
        for doc in loader.load():
            documents_to_load.append(doc)

# Access vector store and create new embeddings if needed

embeddings = OpenAIEmbeddings()
if len(documents_to_load) > 0:
    # see https://github.com/hwchase17/langchain/issues/1310 for using chunk size to help with Rate Limit errors
    print_verbose("RecursiveCharacterTextSplitter")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print_verbose("text_splitter.split_documents")
    texts = text_splitter.split_documents(documents_to_load)
    print_verbose("Chroma.from_documents")
    vectordb = Chroma.from_documents(texts, embeddings, client=chroma_client)
else:
    print_verbose("Chroma")
    vectordb = Chroma(embedding_function=embeddings, client=chroma_client)

# Ask questions

if args.query:
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    for q in args.query:
        print_verbose(f"running query: {q}")
        answer = qa.run(q)
        print_verbose("the answer is:")
        print(answer)
