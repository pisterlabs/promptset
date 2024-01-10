import environment

# https://python.langchain.com/en/latest/ecosystem/llamacpp.html
# pip uninstall -y langchain
# pip install --upgrade git+https://github.com/hwchase17/langchain.git
#
# https://abetlen.github.io/llama-cpp-python/
# pip uninstall -y llama-cpp-python
# pip install --upgrade llama-cpp-python
# pip install chromadb
#
# how to create one https://github.com/nomic-ai/pyllamacpp

import os
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

GPT4ALL_MODEL_PATH = '/Users/loctv/Documents/gpt4all/chat/ggml-gpt4all-l13b-snoozy.bin'

def ask(question, qa):
    print('\n' + question)
    print(qa.run(question)+'\n\n')

persist_directory = './.chroma'
collection_name = 'data'
document_name = './documents/state_of_the_union.txt'

llama_embeddings = LlamaCppEmbeddings(model_path=GPT4ALL_MODEL_PATH)

if not os.path.isdir(persist_directory):
    print('Parsing ' + document_name)
    loader = TextLoader(document_name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts, embedding=llama_embeddings, collection_name=collection_name, persist_directory=persist_directory)
    vectordb.persist()
    print(vectordb)
    print('Saved to ' + persist_directory)
else:
    print('Loading ' + persist_directory)
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=llama_embeddings, collection_name=collection_name)
    print(vectordb)

llm = LlamaCpp(model_path=GPT4ALL_MODEL_PATH)

# from langchain.llms import GPT4All
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# callbacks = [StreamingStdOutCallbackHandler()]
# llm = GPT4All(model=GPT4ALL_MODEL_PATH, callbacks=callbacks, verbose=True)


qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 1}))

ask("What did the president say about Kentaji Brown Jackson", qa);
# ask("Question2", qa);
# ask("Question3", qa);