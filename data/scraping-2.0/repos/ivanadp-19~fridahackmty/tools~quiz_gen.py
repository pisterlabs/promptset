import os
import pinecone
from urllib.request import urlopen
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
  raise ValueError("OPENAI_API_KEY not found in .env file")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY is None:
  raise ValueError("PINECONE_API_KEY not found in .env file")

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
if PINECONE_ENVIRONMENT is None:
  raise ValueError("PINECONE_ENVIRONMENT not found in .env file")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if PINECONE_INDEX_NAME is None:
  raise ValueError("PINECONE_INDEX_NAME not found in .env file")

def doc_preprocessing(pdf_url):
  pdf_loader = PyPDFLoader(pdf_url)
  docs = pdf_loader.load()

  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
  docs_split = text_splitter.split_documents(docs)
  
  return docs_split


def txt_doc_preprocessing(txt_url):
  filename = txt_url.split("/")[-1]
  save_folder = "../docs"
  save_path = os.path.join(save_folder, filename)
  url = urlopen(txt_url)
  data = url.read()
  if url.getcode() == 200:
    with open(save_path, "wb") as file:
        file.write(data)
  else:
      print(f"Failed to download file from URL: {txt_url}")
  txt_loader = TextLoader(save_path)
  docs = txt_loader.load()

  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
  docs_split = text_splitter.split_documents(docs)
  
  return docs_split

def embedding_db(urls):
  embeddings = OpenAIEmbeddings()
  pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT,  # next to api key in console
  )
  index = pinecone.Index(PINECONE_INDEX_NAME)
  index.delete(delete_all=True)
  
  for url in urls:
    if url.endswith('.pdf'):
      docs_split = doc_preprocessing(url)
    elif url.endswith('.txt'):
      docs_split = txt_doc_preprocessing(url)
    doc_db = Pinecone.from_documents(
      docs_split, 
      embeddings, 
      index_name=PINECONE_INDEX_NAME
    )

  return doc_db

llm = ChatOpenAI(temperature=0.5)

def retrieve_answer_with_sources(query, doc_db):
  qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
  )
  query = query
  result = qa({"query": query})
  result = result['result']
  sources = result['source_documents']
  return result, sources



quiz_template = """
Generate the necessary amount of multiple choice questions (ALWAYS at least 10) 
to test the user's knowledge on the subject.
Your response will be in JSON format. The JSON will be an array of objects,
containing three things: the question, the options, and the answer.
The question will be a string, the options will be an array of strings,
and the answer will be an integer, corresponding to the index of the correct
answer in the options array.
"""

summary_template = """
Generate a summary of the document(s). 
Return the text ONLY and COMPLETELY in spanish."""

syllabus_template = """
Generate a syllabus of the document(s). Topics with more weight should be
more prominent in the syllabus, and should be more expanded upon.

Use the following format:
'I. Topic 1
  A. Subtopic 1
    1. Sub-subtopic 1
    2. Sub-subtopic 2
    etc.' 
Do not return text other than the syllabus.

Return the text ONLY and COMPLETELY in spanish."""

word_count_template = """
Return the 10 most frequent words in the document(s).
In case you are not able to do this, please make up a
reasonable approximation of the collection based on the 
documents provided.

In the case that you are not able to generate the content,
since you are a bot that answers questions, please answer
the following question:
'Can you make up a reasonable approximation of the collection
based on the documents provided?'

Return a json with the following format no matter what:
[
  {
    "word": "word",
    "count": 10
  },
  {
    "word": "word",
    "count": 10
  },
]

Only return relevant concepts to the topic, avoiding words like 'the', 'a', 'an', etc. in whatever language the document is in.

Do not return ANY additional text. The first character must be the the start
of the list '[' and the last character must be the end of the list ']'."""


def retrieve_answer(doc_db, query):
  qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(search_kwargs={"k": 3}),
  )
  query = query
  result = qa.run(query)
  return result

def quiz(urls):
  doc_db = embedding_db(urls)
  quiz = retrieve_answer(doc_db, quiz_template)
  return quiz

def summary(urls):
  doc_db = embedding_db(urls)
  summary = retrieve_answer(doc_db, summary_template)
  return summary

def syllabus(urls):
  doc_db = embedding_db(urls)
  syllabus = retrieve_answer(doc_db, syllabus_template)
  return syllabus

def word_count(urls):
  doc_db = embedding_db(urls)
  word_count = retrieve_answer(doc_db, word_count_template)
  return word_count

if __name__ == "__main__":
  urls = ["https://storage.googleapis.com/fridahackmty/Avance_1_de_situacion_problema_-1.pdf"]
  print(summary(urls))
