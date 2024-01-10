import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunking(text):
  rec_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap = 200,
    length_function = len,
  )
  chunks = rec_text_splitter.split_text(text)
  return chunks

