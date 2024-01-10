from pdf_metadata import get_pdf_metadata
from pdf_metadata_llm import get_pdf_metadata_using_llm

def get_pdf_documents(pdf_files):
  from langchain.document_loaders import PyMuPDFLoader,DirectoryLoader,UnstructuredPDFLoader

  docs =[]

  import re

  for pdf_fullpath in pdf_files:
    metadata = get_pdf_metadata(pdf_fullpath)

    if metadata != 'None':
      doc = PyMuPDFLoader(pdf_fullpath).load()
      for element in doc:
        element.metadata = metadata
        element.page_content = re.sub('\n+',' ',element.page_content.strip())
        docs.append(element)
  
    else:
      doc = PyMuPDFLoader(pdf_fullpath).load()
      print(f"{pdf_fullpath} is not identified! Using other strategy!!")
      metadata = get_pdf_metadata_using_llm(doc)
      if metadata != 'None':
        for element in doc:
          element.metadata = metadata
      for element in doc:
        element.page_content = re.sub('\n+',' ',element.page_content.strip())
        docs.append(element)
  
  return docs