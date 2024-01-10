import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY")
import pinecone
import time

from uuid import uuid4
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override=True)  # read local .env file

pinecone_api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

path_tutorial_docs = "../../doc_loader/files/docs2/zkapps/tutorials"

files = os.listdir(path_tutorial_docs)

md_files = [file for file in os.listdir(path_tutorial_docs) if file.endswith(".md")]
docs = DirectoryLoader(
    path_tutorial_docs + "/", glob="**/*.md", loader_cls=TextLoader, show_progress=True
).load()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splitted_docs = [
    markdown_splitter.split_text(doc.page_content) for doc in docs
]

# Char-level splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

# SPLITTING
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)

splitted_docs = [text_splitter.split_documents(doc) for doc in md_header_splitted_docs]

# EMBEDDING
model_name = "text-embedding-ada-002"
texts = [t.page_content for c in splitted_docs for t in c]


def dict_to_list_of_strings(input_dict):
    result = []
    for key, value in input_dict.items():
        result.append(f"{key}: {value}")
    return result


metadatas = [t.metadata for c in splitted_docs for t in c]
string_metadatas = [" ".join(dict_to_list_of_strings(i)) for i in metadatas]
print("Created", len(string_metadatas), "texts")

chunks = [
    string_metadatas[
        i : (i + 1000) if (i + 1000) < len(string_metadatas) else len(string_metadatas)
    ]
    for i in range(0, len(string_metadatas), 1000)
]
embeds = []

print("Metadatas length: ", len(string_metadatas))

print("Have", len(chunks), "chunks")
print("Last chunk has", len(chunks[-1]), "texts")

for chunk, i in zip(chunks, range(len(chunks))):
    chunk = [i.replace("", "NONE") if i == "" else i for i in chunk]
    print("Chunk", i, "of", len(chunk))
    new_embeddings = client.embeddings.create(input=chunk, model=model_name)
    new_embeds = [emb.embedding for emb in new_embeddings.data]

    embeds.extend(new_embeds)

print("Embeds length: ", len(embeds))

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "zkappumstad"
index = pinecone.Index(index_name)

ids = [str(uuid4()) for _ in range(len(texts))]

vector_type = os.getenv("CODE_VECTOR_TYPE") or "CODE_VECTOR_TYPE"

vectors = [
    (
        ids[i],
        embeds[i],
        {
            "text": texts[i],
            "title": dict_to_list_of_strings(metadatas[i]),
            "vector_type": vector_type,
        },
    )
    for i in range(len(texts))
]

for i in range(0, len(vectors), 100):
    batch = vectors[i : i + 100]
    print("Upserting batch", i)
    index.upsert(batch)

time.sleep(5)
print(index.describe_index_stats())
print("Tutorial Loader completed!")