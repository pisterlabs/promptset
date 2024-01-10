from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import io

load_dotenv()


# print(len(string))

# Data for AgilityModelInfoVectorStore -> corpus_info.txt
# Data for AgilityModelVectorStore -> corpus.txt


corpus_file = "corpus.txt"
vec_name = "AgilityModelVectorStore"

print(f"Reading File [{corpus_file}]...")
corpus = io.open(corpus_file, "r", encoding="utf-8").read()


# Chunk size for AgilityModelInfoVectorStore -> 2000
# Chunk size for AgilityModelVectorStore -> 700

size = 700
print(f"Splitting Text [chunk-size={size}]...")

# Splitting document in to Text
text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=size, length_function=len)
document = text_splitter.split_text(corpus)

print("Calling OpenAI Embeddings...")
embed = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

print("Converting to vectors..", end="\n")
vecdb = Chroma.from_texts(document, embed, persist_directory=vec_name)

print("Done!")

