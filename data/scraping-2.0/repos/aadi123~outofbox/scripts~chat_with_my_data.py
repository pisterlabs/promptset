#! /usr/bin/env python3

import openai
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain import Query


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]

loader = TextLoader(file_path="/home/aaditya/5_year_plan.md")
docs = loader.load()
txt = " ".join([d.page_content for d in docs])
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=30, separators=["\n\n", "\n", "(?<=\. )", " ", ""], is_separator_regex=True
)

trans_docs = r_splitter.split_documents(markdown_splitter.split_text(txt))

# print(trans_docs)

from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
pinecone.init(api_key="8949163c-701c-4d86-8df3-31a757f1abd0", environment="gcp-starter")
pinecone.create_index("langchain-self-retriever-demo", dimension=1536)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.vectorstores import Pinecone

metadata_field_info = [
    AttributeInfo(
        name="Header 1",
        description="The top level header for the page content, should be one of `Career Goals`, `Hobby Goals`, or `Other Goals`",
        type="string",
    ),
    AttributeInfo(
        name="Header 2",
        description="The second level header for the page content, should be one of `Chess Goals`, `Music Goals`, `Finance Goals`, `Time Management Goals`, `Entertainment Goals`, `Videogame Goals`, `Social Goals`, or `Fitness Goals`",
        type="string",
    ),
]

embedding = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(
    trans_docs, embedding, index_name="langchain-self-retriever-demo"
)
from langchain.retrievers.self_query.base import SelfQueryRetriever


from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
document_content_description = "5 year goals"
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info
)
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
while True:
    question = input()
    docs = retriever.get_relevant_documents(question)
    for d in docs:
        print(d.metadata)
    # print(len(docs))
    # print(docs)
    # result = qa_chain({"query": question})
    # print(result["result"])
