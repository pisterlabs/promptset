from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain. embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain. chat_models import ChatOpenAI

loader = PyPDFLoader("/Users/tr/Downloads/数字化转型：可持续的进化历程 - 埃森哲.pdf")
pages = loader. load_and_split()

docsearch = Chroma. from_documents(pages, OpenAIEmbeddings())

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(),chain_type="stuff",retriever=docsearch.as_retriever())
qa.run('这本书的作者是谁？')