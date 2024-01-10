import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader("2023", glob="*.txt")
docs = loader.load_and_split()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=docsearch.as_retriever())
query = "What are the three most important points in the text?"
qa.run(query)

qa2 = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="refine",
                                    retriever=docsearch.as_retriever())
qa2.run(query)