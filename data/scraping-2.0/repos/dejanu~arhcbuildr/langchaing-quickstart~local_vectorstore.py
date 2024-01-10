#!/usr/bin/env python3

import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

file = os.path.join(os.getcwd(), "datasets", "arch_dict.csv")
print(file)
loader = CSVLoader(file_path=file)

# create in memory vectorstore
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

# ready query from stdin
query = input("Intrebarea ta: ")
response = index.query(query)
print(response)
display(Markdown(response))

