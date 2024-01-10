# !pip install langchain
# !pip install faiss-cpu
# !pip install openai

import os
os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI_API_KEY"

chat = "{path_to_chat}"

from langchain.document_loaders import WhatsAppChatLoader
loader = WhatsAppChatLoader("example_data/whatsapp_chat.txt")
data = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n', 
                                      chunk_size=1000, 
                                      chunk_overlap=200)


docs = text_splitter.split_documents(data)

import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# vectorStore_openAI = FAISS.from_documents(docs, embeddings)

# with open("faiss_store_openai.pkl", "wb") as f:
#     pickle.dump(vectorStore_openAI, f)


with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)


from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI


llm=OpenAI(temperature=0, model_name='gpt_3.5_turbo')

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

chain({"question": "{Put text here}"}, return_only_outputs=True)
