import subprocess
import os
from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from tika import parser # pip install tika
from colorama import init, Fore, Back
init()




def tryit(kwargs, arg, default):
  try:
    rs = kwargs[arg]
  except:
    rs = default
  return rs

def prunlive(cmd, **kwargs):
  # print("+++++++++++++",cmd)
  debug = tryit(kwargs, "debug", False)
  dryrun = tryit(kwargs, "dryrun", False)
  # cmd = str(cmd).replace("~","\xC2\xA0")
  if dryrun == "print":
    print(Fore.YELLOW + cmd + Fore.RESET)
    return

  # cmd = cmd.replace("~", "X")
  # cmd = cmd.replace("~", "\u00A0")
  scmd = cmd.split()
  # print("===========", scmd)
  for i in range(len(scmd)):
    scmd[i] = scmd[i].replace("~", " ")
    scmd[i] = scmd[i].replace('"', "")
  if debug:
    print(Fore.YELLOW + cmd + Fore.RESET)
    # pprint(scmd)

  process = subprocess.Popen(scmd, stdout=subprocess.PIPE)
  for line in process.stdout:
    print(Fore.RED)
    sys.stdout.write(line.decode("utf-8"))
    print(Fore.RESET)


def split_path(pstr):
  dirname = os.path.dirname(pstr)

  if dirname == "" or dirname == ".":
    dirname = os.getcwd()
  basename = os.path.basename(pstr)
  ns = basename.split(".")
  ext = ns[-1]
  nameonly = "".join(ns[:-1])
  fullpath = f"{dirname}/{basename}"

  return {
    "dirname": dirname,
    "basename": basename,
    "ext": ext,
    "nameonly": nameonly,
    "fullpath": fullpath,
  }

def loaddata(doc):
  # see https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
  docparts = split_path(doc)
  # if verbose: print("Loading Data:", end="")
  if doc.find("http") != -1:
    # if verbose: print("http")
    loader = WebBaseLoader(doc)
  if docparts['ext'] == "pdf":
    # if verbose: print("pdf")
    loader = PyPDFLoader(doc)


  return loader.load()

def splitdata(data):
  # if verbose: print("Splitting Data")
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
  return text_splitter.split_documents(data)

def embed_data(all_splits):
  # if verbose: print("Embedding Data")
  oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
  return Chroma.from_documents(documents=all_splits, embedding=oembed)

def query_data(question, vectorstore,ollama):
  # if verbose: print("Query Data")
  # question="Who is Neleus and who is in Neleus' family?"
  docs = vectorstore.similarity_search(question)
  qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
  rs = qachain({"query": question})
  return rs

def getContent(fn):
  print(Fore.WHITE + f"{fn}" + Fore.RESET)
  # raw = parser.from_file('/home/jw/store/sites/tholonia/chirpy2/assets/material/Introduction_to_Complex_Systems_Sustainability_and.pdf')
  raw = parser.from_file(fn)
  doc = raw['content']
  return doc
