from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from revChatGPT.V1 import Chatbot
import yaml
import openai

markdown_files = ["docs/Player_customization/Create_a_wearable.md"]

for markdown_file in markdown_files:
    with open(markdown_file, 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 16000 * 0.8,
        chunk_overlap  = 0,
    )

    texts = text_splitter.create_documents([markdown_content * 10])
    print("len:", len(texts))
    print(texts[0].page_content)
    print(texts[1].page_content)