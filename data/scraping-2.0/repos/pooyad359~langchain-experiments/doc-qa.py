import re
import sys
from pathlib import Path

import openai
import tiktoken
from dotenv import load_dotenv
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai.api_resources import engine
from pqdm.threads import pqdm
from rich.console import Console

load_dotenv()
MODEL_NAME = "gpt-3.5-turbo-16k"
console = Console()
llm = OpenAI(temperature=0)
console.print("Using model", llm.model_name, style="bold green")
text_splitter = CharacterTextSplitter()
CHUNK_SIZE = 1500
OVERLAP = 200


def process_document(file: Path):
    content = file.read_text()
    console.print("Processing", file, style="bold yellow")
    console.print("Number of characters:", len(content), style="bold green")

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        length_function=len,
    )
    texts = splitter.split_text(content)
    console.print("Number of chunks:", len(texts), style="bold green")
    return texts


class DocSearch:
    def __init__(self, texts):
        self.texts = texts
        self.embeddings = OpenAIEmbeddings(model=llm.model_name)
        self.doc_search = FAISS.from_texts(texts, self.embeddings)

    def _similarity_search(self, query):
        docs = self.doc_search.similarity_search(query)
        return docs

    def query(self, query):
        docs = self._similarity_search(query)
        return "Thinking..."


def main(file: Path):
    texts = process_document(file)
    doc_search = DocSearch(texts)
    console.print("Ready to search. Go ahead and ask a question.", style="bold green")
    console.print("(Type `q` to exit)", style="bold green")
    query = ""
    while True:
        query = console.input(">> ")
        if not query:
            continue
        elif query.lower() == "q":
            break
        response = doc_search.query(query)
        console.print(response, style="bold green")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
