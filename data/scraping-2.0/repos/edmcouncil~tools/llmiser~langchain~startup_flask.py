from time import time

from flask import Flask, render_template, request, jsonify
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS

from llmiser.langchain.config import EMBEDDING_MODEL
from llmiser.langchain.store_creator import create_and_persist_store

app = Flask(__name__)

def process_question(question) -> tuple:
    answer = retrieval_qa({"query": question})
    return answer['result']


@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        print('step1')
        prompt = request.form['prompt']
        response = process_question(prompt)
        print(response)
        return jsonify({'response': response})
    return render_template('index.html')


def setup_retrieval_qa():
    vectorstore = FAISS.load_local(folder_path='index/', embeddings=EMBEDDING_MODEL)
    # llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, max_tokens=4096)
    llm = Bedrock(model_id="anthropic.claude-v2")
    template = \
        """You are now a conversational assistant whose task is to answer questions about financial concepts only on the basis of the FIBO ontology.
        FIBO ontology is available as a set of 161 data dictionary files.
        Answer the questions only using the context.
        Context: {context}
        Question: {question}
        Answer:"""
    chain_prompt = PromptTemplate.from_template(template)
    
    global retrieval_qa
    retrieval_qa = \
        RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 48}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': chain_prompt})
    
    
if __name__ == '__main__':
    # create_and_persist_store(input_docs_folder_path='../gpt_files/', store_path='index/')
    setup_retrieval_qa()
    app.run(debug=True)
