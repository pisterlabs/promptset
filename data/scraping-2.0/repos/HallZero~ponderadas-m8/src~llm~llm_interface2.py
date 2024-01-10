import requests
import json
import gradio as gr
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import torch

torch.cuda.empty_cache() 

loader = TextLoader("./safety_consideration.txt")
document = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    length_function=len,
)
docs = text_splitter.split_documents(document)

for doc in docs:
    print(doc)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(docs, embedding_function)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = Ollama(model="security")

chain = ( {"context": retriever, "question": RunnablePassthrough()} | prompt | model)

def generate_response(prompt):
    print(prompt)
    msg = ""
    for s in chain.stream(prompt):
        print(s, end="", flush=True)
        msg += s
        if '<|im_end|>' in msg:
            break
        yield msg

iface = gr.Interface(
  fn=generate_response,
  inputs="text",
  outputs="text")

iface.launch()