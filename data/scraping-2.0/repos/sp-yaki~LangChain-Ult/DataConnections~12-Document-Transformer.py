from langchain.document_loaders import HNLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

# loader = HNLoader('https://news.ycombinator.com/item?id=30084169')
# data = loader.load()
# print(data[0].page_content)

with open('some_data/some_report.txt') as file:
    speech_text = file.read()

text_splitter = CharacterTextSplitter(separator="\n\n",chunk_size=1000) #1000 is default value
texts = text_splitter.create_documents([speech_text])

print(texts)
print('-----------------------\n')
print(texts[0])
print('-----------------------\n')
print(len(texts[0].page_content))
print('-----------------------\n')
print(len(texts[1].page_content))
print('-----------------------\n')

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = 500)
# texts = text_splitter.split_text(data[0].page_content)

# print(texts)