from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
import spacy
f = open('G:\\Work\\Documents\\OUTPUT\\401(k) Loan_ Should You Borrow Money From Your 401(k)_.txt', 'r', encoding='utf-8')
parsed = f.read()
f.close()

parsed = parsed.replace("\n\n"," ")

text_splitter = CharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2048,
    chunk_overlap = 128,
    length_function = len,
    separator = " ",
)

text_rec_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2048,
    chunk_overlap = 128,
    length_function = len,
    separators=[" "]
)
texts = text_splitter.create_documents([parsed])
print(texts)

texts_rec = text_rec_splitter.create_documents([parsed])
print(texts_rec)

text_splitter_tiktoken = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2048, chunk_overlap=64, separator=""
)
texts_tiktoken = text_splitter_tiktoken.create_documents([parsed])
print(texts_tiktoken)

text_splitter_rec_tiktoken = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=16, separators=[" "]
)
texts_rec_tiktoken = text_splitter_rec_tiktoken.create_documents([parsed])
print(texts_rec_tiktoken)

sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
count_start_and_stop_tokens = 2
text_token_count = sentence_splitter.count_tokens(text=parsed) - count_start_and_stop_tokens
token_multiplier = sentence_splitter.maximum_tokens_per_chunk // text_token_count + 1
text_to_split = parsed * token_multiplier
texts_sentence = sentence_splitter.create_documents([text_to_split])
print(texts_sentence)
