import streamlit as st
import os 
from langchain.document_loaders import ConfluenceLoader
import openai
from bs4 import BeautifulSoup
import nltk
#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
import openai
import numpy as np
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader

openai.api_key = ''
os.environ['OPENAI_API_KEY'] = ''

loader = ConfluenceLoader(url="", token="")
documents = loader.load(
    space_key="", include_attachments=False, limit=50, max_pages=50)

# for page in documents:
#         title = page.metadata['title']
#         htmlbody = page.page_content
#         htmlParse = BeautifulSoup(htmlbody, 'html.parser')
#         body = []
#         for para in htmlParse.find_all("p"):
#             sentence = para.get_text()
#             tokens = nltk.tokenize.word_tokenize(sentence)
#             token_tags = nltk.pos_tag(tokens)
#             tags = [x[1] for x in token_tags]
#             if any([x[:2] == 'VB' for x in tags]): # There is at least one verb
#                 if any([x[:2] == 'NN' for x in tags]): # There is at least noun
#                     body.append(sentence)
#         body = '. '.join(body)

from langchain.embeddings import OpenAIEmbeddings
OpenAIEmbeddings.openai_api_key = ''

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(documents[0].page_content)

qdrant = Qdrant.from_documents(
    documents,
    embeddings_model,
    location=":memory:",  
    collection_name="my_documents",
)

query = "related enabled calculation"
found_docs = qdrant.similarity_search(query)

print(found_docs[0].page_content)

# messages = [
#     {"role": "system", "content": "You are a helpful assistant that summarizes text."},
#     {"role": "assistant", "content": documents[0].page_content},
# ]

# response = openai.ChatCompletion.create(
#     model='gpt-3.5-turbo',
#     messages=messages,
#     max_tokens=50, 
# )

# summary = response['choices'][0]['message']['content']
# #//summary = response.choices[0].text.strip()

# # print(summary);

# st.write(summary)