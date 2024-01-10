from flask import Flask, request, jsonify, render_template, send_from_directory#from sklearn.feature_extraction.text import TfidfVectorizer
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from pathlib import Path
import dotenv
import openai

dotenv.load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')#openai.api_key = openai_api_key

file_paths = [
    str(Path("static/docs/(CV) JuanD Abril EN (1).pdf")),
    str(Path("static/docs/(2023) [CV] Juan D Abril B (1) (1) (1).pdf")),
]
files_folder = str(Path("static/docs"))
#print(file_path)
persist_directory = "./storage"

loader = PyPDFDirectoryLoader(files_folder)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name='gpt-3.5-turbo')

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

###while True:
##        user_input = input("Enter a query: ")
##        if user_input == "exit":
#            break

#        query = f"###Prompt {user_input}"
#        try:
#            llm_response = qa(query)
#            print(llm_response["result"])
#        except Exception as err:
#            print('Exception occurred. Please try again', str(err))

#@app.route('/', methods=['GET'])
#
#def
#index():
#return render_template('index.html')


# Inicializar la aplicación Flask
app = Flask(__name__)
# Cargar el modelo previamente entrenado y el vectorizador
#with open('/modelo_nb.pkl', 'rb') as file:
#modelo = pickle.load(file)

#with open('/tfidf_vectorizer.pkl', 'rb') as file:
#vectorizer = pickle.load(file)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process-input', methods=['POST'])
def process_input():
    user_input = request.form['userInput']    #user_input = input("Enter a query: ")
    query = f"Please answer based on the document content only ###Prompt {user_input}"
    try:
        llm_response = qa(query)
        print(llm_response["result"])
    except Exception as err:
        print('Exception occurred. Please try again', str(err))

    processed_text = r"<br>"
    processed_text += f"<b>{user_input}</b>"
    processed_text += r"<br><hr>"
    processed_text += llm_response["result"]#"You entered: " + user_input
    return render_template('index.html', processed_text=processed_text)

@app.route("/download")
def download():
    return send_from_directory("static\docs\CV.pdf", mimetype="application/pdf")

#@app.route("/style")
#def style():
#    return send_file("style.css", mimetype="text/css")

#@app.route("/script")
#def script():
#    return send_file("script.js", mimetype="application/javascript")


@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el dato del usuario
    data = request.json
    question = data['review']#    question = query

    # Convertir el texto a una matriz TF-IDF
    #review_vectorized = vectorizer.transform([review])

    # Realizar la predicción
    #prediction = modelo.predict(review_vectorized)[0]
    #probability = max(modelo.predict_proba(review_vectorized)[0])

    #return jsonify({'prediction': prediction, 'probability': probability})
    return jsonify({'message': 'hola mundo'})


if __name__ == '__main__':
    app.run(debug=True)