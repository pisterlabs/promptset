import os

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import requests


def load_files(directory):
    files_contents = []
    for filename in os.listdir(directory):
        if filename.endswith(".tex"):
            with open(os.path.join(directory, filename), "r") as file:
                file_content = file.read()
                files_contents.append(
                    ({"source": filename, "filename": filename}, file_content)
                )
    return files_contents


sources = load_files("papers_db")

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for (metadata, source) in sources:
    for chunk in splitter.split_text(source):
        source_chunks.append(Document(page_content=chunk, metadata=metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

chain = load_qa_with_sources_chain(OpenAI(temperature=0))


def print_answer(question):
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=False,
        )["output_text"]
    )
    print()


print_answer("what is the most interesting thing in these papers?")

print_answer("what is the most interesting impactful paper here and why?")

print_answer("Summarize the top 5 points from these papers")
