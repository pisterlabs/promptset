#! /usr/bin/python
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import pickle
import json



chain = load_qa_with_sources_chain(OpenAI(temperature=0))

def print_answer(question):
  print(f"QUESTION: {question}")
  
  with open("search_index.pickle", "rb") as f:
    search_index = pickle.load(f)
  result = chain(
      {
        "input_documents": search_index.similarity_search(question, k=4),
        "question": question,
      },
      return_only_outputs=True,
    )
  print("ANSWER:")
  print( result["output_text"] )

print_answer("does armstrong have bandwidth caps?")