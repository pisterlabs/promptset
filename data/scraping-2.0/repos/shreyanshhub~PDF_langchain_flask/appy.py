import os
from flask import Flask, render_template, request, redirect, url_for, session
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_token', methods=['POST'])
def set_token():
    if request.method == 'POST':
        api_token = request.form.get('api_token')
        if api_token:
            session['api_token'] = api_token  # Store the token in the session
            return redirect(url_for('ask_questions'))

    return redirect(url_for('index'))

@app.route('/ask_questions')
def ask_questions():
    if 'api_token' in session:
        return render_template('ask_questions.html')
    else:
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'api_token' not in session:
            return redirect(url_for('index'))

        pdf_file = request.files['pdf_file']
        if pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings()

            knowledge_base = FAISS.from_texts(chunks, embeddings)

            user_question = request.form.get('user_question')
            if user_question:
                api_token = session.get('api_token')  # Retrieve the stored API token
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 5, "max_length": 64},
                    huggingfacehub_api_token=api_token  # Use the stored API token
                )
                docs = knowledge_base.similarity_search(user_question)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

                # Store the PDF content in the session only if it hasn't been stored before
                if 'pdf_content' not in session:
                    session['pdf_content'] = text

                return render_template('ask_questions.html', response=response)

    return redirect(url_for('ask_questions'))

if __name__ == '__main__':
    app.run(debug=True)


