from dotenv import load_dotenv
from flask import Flask, Response
from flask_cors import CORS 
import time
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms.google_palm import GooglePalm
from langchain.vectorstores.chroma import Chroma
from langchain.chains import VectorDBQA
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def read_files_in_folder():
    data = []
    sources = []
    file_count=0

    if not os.path.exists("./data/files"):
        print(f"The folder ./data does not exist.")
        return data, sources, file_count

    for filename in os.listdir("./data/files"):
        file_path = os.path.join("./data/files/", filename)

        if filename == "file_count.txt":
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                file_count = int(file_content)
            continue

        if os.path.isfile(file_path) and filename.endswith(".txt"):
            sources.append(filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                data.append(file_content)

    return data, sources, file_count

def create_vector_db(text_chunks):
    embeddings = GooglePalmEmbeddings(google_api_key=os.environ['API_KEY'])
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory="./data/vector_store")
    vector_store.persist()
    return vector_store

def get_vector_db():
    embeddings = GooglePalmEmbeddings(google_api_key=os.environ['API_KEY'])
    vector_store = Chroma(persist_directory="./data/vector_store",embedding_function=embeddings)
    return vector_store


@app.route('/api/askme/<question>')
def ask_me(question):
    # load_dotenv(dotenv_path="./.envfile")
    load_dotenv()
    def generate():
        data, sources, count = read_files_in_folder()
        if count < len(sources):
            store = create_vector_db(data)
            with open("./data/files/file_count.txt", 'w', encoding='utf-8') as file:
                file.write(str(len(sources)))
        else:
            store = get_vector_db()
        QA = VectorDBQA.from_chain_type(llm=GooglePalm(google_api_key=os.environ['API_KEY']), chain_type="stuff", vectorstore=store)
        answer = QA({'query': question}, return_only_outputs=True)
        words = answer['result'].split()
        for word in words:
            time.sleep(0.1)
            yield f"data: {word}\n\n"
    return Response(generate(), content_type='text/event-stream')


if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0",port=port, debug=True)