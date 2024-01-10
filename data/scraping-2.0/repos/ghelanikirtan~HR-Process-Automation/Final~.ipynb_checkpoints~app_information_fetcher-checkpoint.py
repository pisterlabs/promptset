import os, re, json, requests, shutil
import numpy as np, pandas as pd
import sqlite3
from flask_cors import CORS

from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


from flask import Flask, request, jsonify
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Document Loader:
# path_to_cv = os.path.join('Final','samples_resume',os.listdir('samples_resume')[0])
# path_to_cv

# def load_cv(path_to_cv):
    # loader = PyMuPDFLoader(path_to_cv)
    # document = loader.load()
    # return document


## Embeddings setup and persist_directory path setup for chromaDB:
# embeddings = OpenAIEmbeddings()
# persist_dir = os.path.join('app_persist_directory')


app = Flask(__name__)
CORS(app)

# Database Connect:
DATABASE_path = os.path.join('Final','database', 'database_fetch.db')
print("Database Path: ", DATABASE_path)

def get_db():
    db = getattr(Flask, '_database', None)
    if db is None:
        db = Flask._database = sqlite3.connect(DATABASE_path)
    return db

# @app.teardown_appcontext
# def close_connection(exception):
#     db = getattr(Flask, '_database', None)
#     if db is not None:
#         db.close()        

@app.route('/')
def index():
    conn = sqlite3.connect(DATABASE_path)
    cur = get_db().cursor()
    # print(cur, "this is cursor")
    cur.execute("SELECT * FROM applicants")
    applicants = cur.fetchall()
    
    conn.close()
    return str(applicants)


@app.route('/add', methods=['POST'])
def add_details():
    data = request.get_json()
    
    id = np.random.randint(0,10000)
    name = data['name']
    contact_no = data['contact_no']
    email = data['email']
    skills = data['skills']
    job_role = data['job_role']

    resume_file = "Temptesting"
    print(data)
    print(f"{name}\n{contact_no}\n{email}\n{skills}\n{job_role}\n")
    conn = sqlite3.connect(DATABASE_path)
    cur = conn.cursor()
    
    cur.execute(f"INSERT INTO applicants (name, email, contact_no, job_role, resume_file, skills) VALUES (?,?,?,?,?,?)",(name,email,contact_no,job_role,resume_file,skills))
    id +=1
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Congratulations You have applied for the following role successfully!!"})

if __name__ == "__main__":
    app.run(debug=True)





# Flask Application Building:
