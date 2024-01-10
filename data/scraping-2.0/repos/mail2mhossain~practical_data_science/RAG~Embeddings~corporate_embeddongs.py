import os
import glob
import textwrap
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader  # for textfiles
from langchain.text_splitter import CharacterTextSplitter  # text splitter
from langchain.embeddings import HuggingFaceEmbeddings  # for using HugginFace models

# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import (
    FAISS,
)  # facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader  # load pdf
from langchain.indexes import (
    VectorstoreIndexCreator,
)  # vectorize db index with chromadb
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    UnstructuredURLLoader,
)  # load urls into docoument-loader

load_dotenv("../.env")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Document Loader
loader = TextLoader("../Documents/KS-all-info_rev1.txt")
documents = loader.load()


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
docs = text_splitter.split_documents(documents)


# Embeddings
embeddings = HuggingFaceEmbeddings()

# Create the vectorized db
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
db = FAISS.from_documents(docs, embeddings)

# query = "What is Hierarchy 4.0?"
# docs = db.similarity_search(query)

# print(wrap_text_preserve_newlines(str(docs[0].page_content)))

pdf_folder_path = "../Documents/"
os.listdir(pdf_folder_path)
pdf_files = glob.glob(os.path.join(pdf_folder_path, f"*{'.pdf'}"))

# print(len(pdf_files))

loaders = [UnstructuredPDFLoader(fn) for fn in pdf_files]
# print(len(loaders))
index = VectorstoreIndexCreator(
    embedding=embeddings,
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
).from_loaders(loaders)

llm6 = HuggingFaceHub(
    repo_id="MBZUAI/LaMini-Flan-T5-783M",
    model_kwargs={"temperature": 0, "max_length": 512},
)

chain = RetrievalQA.from_chain_type(
    llm=llm6,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key="question",
)

chain.run("What is the difference between a PLC and a PC?")

chain.run("What is a PLC?")
