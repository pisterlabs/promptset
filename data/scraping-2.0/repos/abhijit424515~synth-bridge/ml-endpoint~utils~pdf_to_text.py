import os
from functools import reduce
from langchain.document_loaders import PyPDFLoader

# Extracts text from PDF binary data
def get_text(id, data):
  path = f"temp/{id}.pdf"
  open(path, "wb").write(data)
  loader = PyPDFLoader(path) # extract the text data from pdf, and concatenate page-wise texts
  text = reduce(lambda p,q: p+" "+q ,[x.page_content for x in loader.load_and_split()], "")
  os.remove(path)
  return text
