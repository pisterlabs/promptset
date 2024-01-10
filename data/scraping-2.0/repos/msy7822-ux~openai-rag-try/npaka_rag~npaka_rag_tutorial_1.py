## URL: https://note.com/npaka/n/nf2849b26a524

import os
import openai

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader("./npaka_rag/bocchi.txt")
documents = loader.load()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
retriever = Chroma.from_documents(texts, embeddings).as_retriever()

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# RetrievalQAチェーンの生成
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

print(qa.run("未確認ライオットに参加するために必要なことは何ですか？"))
