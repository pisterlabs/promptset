import datetime

import chainlit as cl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from llama_index import Document, PineconeReader

from api_gpt.index.pinecone_index import *

# email_documents = get_email_documents(
#     user_id="lizhenpi@gmail.com",
#     email_query="category:primary is:unread after:"
#     + (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y/%m/%d"),
# )
# print(f"email_documents : {email_documents[0]}")

documents = [
    Document(
        text="OpenAI (Open Artificial Intelligence) is an artificial intelligence research laboratory and company.",
        metadata={"user_id": "testuser"},
    ),
    Document(
        text="Google is a multinational technology company that specializes in various internet-related services and products.",
        metadata={"user_id": "testuser"},
    ),
]
new_document = Document(
    text="Roblox is an online platform and game creation system that allows users to design, create, and play a wide variety of games and experiences. ",
    metadata={"user_id": "testuser"},
)


index = upsert_documents(user_id="testuser", documents=documents)
nodes = retrieve(user_id="testuser", query="What is Google?")

assert len(nodes) == 2
assert (
    nodes[0].node.text
    == "Google is a multinational technology company that specializes in various internet-related services and products."
)

delete(user_id="testuser")
nodes = retrieve(user_id="testuser", query="What is Google?")
assert len(nodes) == 0
