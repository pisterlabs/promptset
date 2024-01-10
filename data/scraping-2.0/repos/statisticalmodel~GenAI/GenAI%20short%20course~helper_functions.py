import sys
from typing import List, Optional, Type

# NLP modules
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,JSONLoader
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
import wikipedia
import fitz



# Helper functions for internet search using Wikipedia and DuckDuckGo
#-------------------------------------------------------------------------------
def wiki_search_api(query, how_many=800):
    ny = wikipedia.page(query, auto_suggest=True)
    print(ny.url)
    result=ny.content
    return result[:how_many]


def find_keyword(query):
    if 'keyword' in query:
        index=query.find('keyword')+9
        temp=query[index:]
        if temp.endswith('.'):
            temp=temp[:-1]
        else:
            temp=temp
    return temp


def get_search_results_ddg(query: str, max_results: int=10):
    '''Get search results using DuckDuckGo API'''
    ddg_search_result = DuckDuckGoSearchRun(max_results=max_results).run(query)
    return ddg_search_result




# Helper functions for prompting
#-------------------------------------------------------------------------------
def get_contexts(pdf: str, query: str, faiss_embeddings_dict: dict, nr_hits: int=5):
    contexts = []
    faiss_index = faiss_embeddings_dict[pdf]
    top_hits = similarity_search_FAISS(search_query=query, nr_hits=nr_hits, index_store=faiss_index)
    context = "".join([document.page_content for document in top_hits])
    contexts.append(context)
    
    return contexts


# Helper functions for text retrieval from documents
#-------------------------------------------------------------------------------
def text_from_pdf(file_path: str) -> dict:
  """
  Uses the PyMuPDF library fitz module to extract raw text from a pdf.
    https://pymupdf.readthedocs.io/en/latest/

  Input:
  file_path - path to pdf file for parsing

  Output:
  dict - Returns a dictionary with "page_nr" as key and extracted text as value.
  Additional properties can be made available (see link in comment below).
  """
  doc = fitz.open(file_path)
  pages = {}
  for page in doc:
    # type(page) = <class 'fitz.fitz.Page'>
    # (https://pymupdf.readthedocs.io/en/latest/page.html)
    pages[f"page_{page.number}"] = page.get_text()
  return pages

def pdf_dict_to_str(pdf_dicct: dict) -> str:
  """Construct a single text string from text_from_pdf() output"""
  return "\n".join(pdf_dicct.values())


def text_from_txt(
    file_path: str,
    encoding: str=sys.getfilesystemencoding()
    ) -> str:
  """
  Reads a txt file and returns a string object of content

  Input:
  file_path - path to txt file for parsing

  Output:
  str - Returns extracted text as str.
  """
  with open(file_path, encoding=encoding) as f:
    text = f.read()
  return text


# Helper functions for loading text files to langchain Document objects,
# and chunking text/documents into chunks for embedding
#-------------------------------------------------------------------------------
def TextLoader(
    file_path: str,
    loader: Type[PyPDFLoader | JSONLoader | None]=None,
    jq_schema: str='.', # '.content', '.messages[].content'
    content_key='content',
    json_lines: bool=False,
    txt_encoding: str=sys.getfilesystemencoding()
    ) -> List[Document]:
    """
    Converts a pdf, JSON or txt file into a langchain Document object,
    containing methods page_content and metadata.
    Specify type via loader, if not PyPDFLoader | JSONLoader then assumes txt
    """
    pages = None
    if loader==PyPDFLoader:
      loader = PyPDFLoader(file_path=file_path)
      pages = loader.load()   # .load_and_split()
    elif loader==JSONLoader:
      loader = JSONLoader(
          file_path=file_path,
          jq_schema=jq_schema,
          content_key=content_key,
          json_lines=json_lines
          )
      pages = loader.load()
    else:
      text_string = text_from_txt(file_path=file_path, encoding=txt_encoding)
      pages = [Document(page_content=text_string, metadata={"source": file_path})]
    return pages


def chunk_documents(
    documents: List[Document],
    TextSplitter: Type[CharacterTextSplitter | RecursiveCharacterTextSplitter],
    chunk_size: int=512,
    chunk_overlap: int=20,
    separator: str=None,
    ) -> List[Document]:
    """
    TBW ...
    Output: A list of langchain.schema.document.Document objects.
    These have methods
      - Document.page_content [str] contains the text chunk
      - Doumnent.metadata [dict] contains optional metadata
    """
    docs = None
    # CharacterTextSplitter
    if separator!=None:
      text_splitter = CharacterTextSplitter(
      separator = separator,
      chunk_size = chunk_size,
      chunk_overlap  = chunk_overlap
      )
      docs = text_splitter.split_documents(documents)
    #RecursiveCharacterTextSplitter
    else:
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = chunk_size,
          chunk_overlap  = chunk_overlap
      )
      docs = text_splitter.split_documents(documents)

    if docs:
      return docs
    else:
      print(f"Error: could not chunk text.")
      return None


def chunk_text(
    text: str,
    TextSplitter: Type[CharacterTextSplitter | RecursiveCharacterTextSplitter],
    chunk_size: int=512,
    chunk_overlap: int=20,
    separator: str=None,
    ) -> List[Document]:
    """
    TBW ...
    Output: A list of langchain.schema.document.Document objects.
    These have methods
      - Document.page_content [str] contains the text chunk
      - Doumnent.metadata [dict] contains optional metadata
    """
    docs = None
    # CharacterTextSplitter
    if separator!=None:
      text_splitter = CharacterTextSplitter(
      separator = separator,
      chunk_size = chunk_size,
      chunk_overlap  = chunk_overlap
      )
      docs = text_splitter.create_documents([text])
    #RecursiveCharacterTextSplitter
    else:
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = chunk_size,
          chunk_overlap  = chunk_overlap
      )
      docs = text_splitter.create_documents([text])
    if docs:
      return docs
    else:
      print(f"Error: could not chunk text.")
      return None



# Helper functions for embedding chunked text using various embedding models,
# and performing similarity search using a vector database store
#-------------------------------------------------------------------------------
def doc_embedding(
    embedding_model: str,
    model_kwargs: dict={'device': 'cpu'},
    encode_kwargs: dict={'normalize_embeddings': True},
    cache_folder: Optional[str]=None,
    multi_process: bool=False,
    ) -> HuggingFaceEmbeddings:
  """
  TBW...
  """
  embedder = HuggingFaceEmbeddings(
      model_name = embedding_model,
      model_kwargs = model_kwargs,
      encode_kwargs = encode_kwargs,
      cache_folder = cache_folder,
      multi_process = multi_process
  )
  return embedder


def make_index_FAISS(
    chunked_documents: List[Document],
    embedding: HuggingFaceEmbeddings,
    ) -> List:
  """Use FAISS to perform similarity search ..."""
  faiss_index = FAISS.from_documents(
      documents=chunked_documents,
      embedding=embedding
      )
  return faiss_index


def similarity_search_FAISS(
    search_query: str,
    index_store: FAISS,
    nr_hits: int=5,
    ) -> List:
  """Use FAISS to perform similarity search ..."""
  most_similar = index_store.similarity_search(search_query, k=nr_hits)
  return most_similar


def search_result_is_in_top_5(docs, id):
    cnt=0
    for doc in docs:
        if doc.metadata["page"] == id:
            return True, cnt
        cnt+=1
    return False, None