import argparse
from flask import Flask, request, render_template, redirect
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import VectorDBQA, RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from utils.helper_functions import *
import os


os.environ["OPENAI_API_KEY"] = "your openAI api key"
app = Flask(__name__)

# initialize the embeddings using openAI ada text embedding library
embeddings = OpenAIEmbeddings()

# initialize and read the *.pdf object
texts = process_all_pdfs('document_storage', preprocess_langchain=True) # replace 'directory_path' with your directory

# initialize the FAISS document store using the preprocessed text and initialized embeddings
docsearch = FAISS.from_texts(texts, embeddings)
retriever = docsearch.as_retriever()

prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
       provide the answer in a easy and understandable way.
       if the question is not related to the context, please answer with "I do not have it in my context".

       Context: {context}

       User: {question}
       System: """

qa_prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

#chain_type_kwargs = {"prompt": qa_prompt}

# Create a conversation buffer memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever=retriever, memory=memory, combine_docs_chain_kwargs={'prompt': qa_prompt})

chat_history = []

@app.route('/', methods=['GET'])
def home():
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        query = request.form.get('question')
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))
    return render_template('chat.html', chat_history=chat_history)

def main(args):
    # setup the openAI API key

    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a directory path and an OpenAI API key.')
    parser.add_argument('--directory_path', type=str, help='A directory path.')
    parser.add_argument('--api_key', type=str, help='OpenAI API key.')
    args = parser.parse_args()
    main(args)
