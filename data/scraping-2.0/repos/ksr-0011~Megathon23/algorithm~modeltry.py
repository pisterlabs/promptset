from flask import Flask, request, jsonify
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from flask_cors import CORS
from langchain.vectorstores import Vectara
from langchain.document_loaders import TextLoader
from langchain.embeddings import FakeEmbeddings
import os

app = Flask(__name__)
CORS(app)

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
DATA_PATH = 'data/'
vectara_db = None

os.environ["VECTARA_CUSTOMER_ID"] = '71907852'
os.environ["VECTARA_API_KEY"] = 'zqt_BEk6DDXIwTm6XV4XmHgEZYcixR6SLJeoJ9iFUA'
os.environ["VECTARA_CORPUS_ID"] = '1'

def getdb():
    loader = TextLoader(DATA_PATH + 'out2.txt')
    documents = loader.load()
    global vectara_db
    vectara_db = Vectara.from_documents(documents , embedding=FakeEmbeddings(size=768))


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt):
    # qa = RetrievalQA.from_llm(llm=llm, retriever=vectara_db.as_retriever())
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectara_db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.0
    )
    return llm

# QA Model Function
def qa_bot():
    # db = Vectara.load_local(DB_VECTARA_PATH)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm , qa_prompt)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@app.route('/qa', methods=['POST'])
def qa_endpoint():
    query = request.form['query']
    ans = final_result(query)
    print(ans['result'])
    return jsonify({"result": ans["result"]})

if __name__ == '__main__':
    getdb()
    print("Successfully initialised Db")
    app.run()
