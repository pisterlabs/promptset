#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

# hyper parameters
llm_name="gpt-4-0613"
#llm_name="gpt-4"
#llm_name="gpt-3.5-turbo"
#llm_name="gpt-3.5-turbo-0613"
#llm_name="gpt-3.5-turbo-16k"
embedding_model='text-embedding-ada-002'
page_chunk_size = 1024
max_token_num = 4096
conversation_window_size = 3
conversation_token_num = 1024
conversation_history_type = "window" # token or window
vector_db = None

if __name__ == "__main__":
    if (len(sys.argv) == 1) or (len(sys.argv) > 4):
        print("USAGE: " + sys.argv[0] + " new [<doc_dir> [<db_dir>]]")
        print("USAGE: " + sys.argv[0] + " chat [<db_dir>]")
        print("USAGE: " + sys.argv[0] + " question <db_dir>")
        sys.exit(1)

    mode=sys.argv[1]
    db_dir = "DB"
    doc_dir = "documents"
    ans_dir = "answer"

    if mode == "chat":
        if len(sys.argv) != 2 and len(sys.argv) != 3:
            print("USAGE: " + sys.argv[0] + " chat [<db_dir>]")
            sys.exit(1)
        if len(sys.argv) == 3:
            db_dir = sys.argv[2]
        
    if mode == "question":
        if len(sys.argv) != 4:
            print("USAGE: " + sys.argv[0] + " question <db_dir>")
            sys.exit(1)
        question = sys.argv[2]
        db_dir = sys.argv[3]

    if mode == "new":
        if len(sys.argv) != 2 and len(sys.argv) != 4:
            print("USAGE: " + sys.argv[0] + " new [<doc_dir> [<db_dir>]]")
            sys.exit(1)
        if len(sys.argv) == 4:
            doc_dir=sys.argv[2]
            db_dir = sys.argv[3]
    print("DB_DIR =" + db_dir)
    print("DOC_DIR=" + doc_dir)
else:
    conversation_history_type="window"
    conversation_window_size=0

import os
import numpy as np
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory

def create_db(doc_dir, db_dir, embedding_model, chunk_size):
    pdf_files = [ file for file in os.listdir(doc_dir) if file.endswith(".pdf")]
    json_files = [ file for file in os.listdir(doc_dir) if file.endswith(".json")]
    csv_files = [ file for file in os.listdir(doc_dir) if file.endswith(".csv")]
    pptx_files = [ file for file in os.listdir(doc_dir) if file.endswith(".pptx")]
    url_files = [ file for file in os.listdir(doc_dir) if file.endswith(".url")]
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = chunk_size,
        chunk_overlap = 0,
    )
    files = pdf_files + csv_files + pptx_files + url_files + json_files
    pages = []
    for file in files:
        print("INFO: Loading document=" + file)
        if ".pdf" in file:
            loader = PyPDFLoader(doc_dir + '/' + file)
        elif ".csv" in file:
            loader = CSVLoader(doc_dir + '/' + file)
        elif ".pptx" in file:
            loader = UnstructuredPowerPointLoader(doc_dir + '/' + file)
        elif ".json" in file:
            loader = JSONLoader(file_path= doc_dir + '/' + file, jq_schema='.messages[].content')
        elif ".url" in file:
            with open(doc_dir + '/' + file, 'r') as file:
                urls = file.read().splitlines()
            loader = UnstructuredURLLoader(urls = urls)
        else:
            print("WARNING: Not supported document=" + file)
            continue
        #print("INFO: Spliting document=" + file)
        tmp_pages = loader.load_and_split()
        chanked_pages = text_splitter.split_documents(tmp_pages)
        pages = pages + chanked_pages

    print("INFO: Storing Vector DB:" + db_dir)
    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=db_dir)
    vectorstore.persist()


def load_db(db_dir, llm_name, embedding_model, token_num, history_type, num):
    global vector_db
    print("INFO: Setting up LLM:" + db_dir)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0, 
        model_name=llm_name, 
        max_tokens=token_num)

    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    vector_db = vectorstore
    if (history_type == "window"):
        memory = ConversationBufferWindowMemory(k=num, memory_key="chat_history", return_messages=True)
    else:
        memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=num, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        vectorstore.as_retriever(), 
        memory=memory
        )
    return qa

def load_db_with_type(db_dir):
    global llm_name, max_token_num, conversation_history_type, conversation_window_size, conversation_token_num
    if (conversation_history_type == "window"):
        qa = load_db(db_dir, llm_name, embedding_model, max_token_num, conversation_history_type, conversation_window_size)
    else:
        qa = load_db(db_dir, llm_name, embedding_model, max_token_num, conversation_history_type, conversation_token_num)
    return qa

def embedding(text: str) -> list[float]:
    result = openai.Embedding.create(input=text, model=embedding_model)
    if isinstance(result, dict):
        embedding = result["data"][0]["embedding"]
        return embedding
    return []

def cos_sim(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def calc_similarity(str1, str2):
    try:
        s1 = np.array(embedding(str1))
        s2 = np.array(embedding(str2))
        return cos_sim(s1, s2)
    except Exception as e:
        print("An error occurred:", str(e))
        return None

def similarity_search_with_score(db_dir: str, terms: str, top_k: int):
    #print(f"db_dir={db_dir} terms={terms} embedding_model={embedding_model}")
    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    vector_db = vectorstore
    docs = vector_db.similarity_search_with_score(terms, top_k = top_k)
    #print(str(docs))
    #print(f"content: {docs[0][0].page_content}", f"score: {docs[0][1]}")
    #print(f"content: {docs[1][0].page_content}", f"score: {docs[1][1]}")
    return docs


if __name__ == "__main__":
    if mode == "new":
        _ = create_db(doc_dir, db_dir, embedding_model, page_chunk_size)
    elif mode == "question":
        qa = load_db_with_type(db_dir)
        result = qa({"question": question})
        print(result["answer"])
    else:
        qa = load_db_with_type(db_dir)
        while True:
            query = input("> ")
            if query == 'exit' or query == 'q' or query == "quit":
                print("See you again!")
                sys.exit(0)
            print("Q: " + query)

            result = qa({"question": query})
            print("A: "+result["answer"])

            #docs = vector_db.similarity_search_with_score(query, top_k = 1)
            #print(str(docs))
            #print(f"content: {docs[0][0].page_content}", f"score: {docs[0][1]}")
            #print(f"content: {docs[1][0].page_content}", f"score: {docs[1][1]}")