from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

from langchain.document_loaders import TextLoader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

loader = TextLoader('./lecture_1_transcript.txt', encoding='utf8')
documents = loader.load()

embeddings = HuggingFaceEmbeddings()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

chroma_db = Chroma.from_documents(texts, embeddings)

retriever = chroma_db.as_retriever()

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, verbose=True)

qa.run("What is this class about?")
