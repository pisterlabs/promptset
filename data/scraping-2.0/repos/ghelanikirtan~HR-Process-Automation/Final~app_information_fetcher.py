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

prompts = ["What job role is suitable for this resume? [only job role word: No explanation]", 
         "should I shortlist this resume for the {} job role? [only yes|no answer: no explanation]",
         "top technical skills -list [only wordings separated by commas, no explanation] in given resume",
          "Other possible job role candidate is eligible for? [list separated by commas, only words: no explanation",
         "On the basis of the {} skillsets, generate {} logical based hard level screening questions [hard-level : MCQs], Also provide answers at the very end of the output separated by commas",
          "Summarize this resume in {} words"         
         ]

app = Flask(__name__)
CORS(app)

# Database Connect:
DATABASE_path = os.path.join('database', 'raw_database.db')
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
print(os.listdir())
path_to_cv = os.path.join('01.pdf')
path_to_cv

embeddings = OpenAIEmbeddings()
persist_dir = os.path.join('app_persist_directory')
llm = ChatOpenAI()

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
    # Fetching Details
    data = request.get_json()
    print(data)
    
    # name = data['name']
    name = data.get('name')
    contact_no = data['contact_no']
    email = data['email']
    skills = data['skills']
    projectLinks = data['projectLinks']
    job_role = data['job_role']
    # resume_file = data['resume_file']
    
    # print(f"{name}\n{contact_no}\n{email}\n{skills}\n{job_role}\n")
    
    # eligible = data['eligibility']
    # job_role_predicted = data['job_role_predicted']
    # skills_extracted = data['skills_extracted']

    resume_file = 'Temp_Test'
    loader = PyMuPDFLoader(path_to_cv)
    document = loader.load()

    vectordb = Chroma.from_documents(
        documents = document,
        embedding = embeddings,
        persist_directory = persist_dir
    )
    vectordb.persist()
    


    retriever = vectordb.as_retriever(search_kwargs={'k':3})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    # query = f"###Prompt {prompts[0]}"
    # llm_response = qa(query)
    # job_title_prediction = llm_response['result']

    query = f"###Prompt {prompts[1].format(job_role)}"
    llm_response = qa(query)
    isEligible = llm_response['result']
    print("Eligible:", isEligible)
    query = f"###Prompt {prompts[2]}"
    llm_response = qa(query)
    skills_extracted = llm_response['result']
    print("Skills: ",skills_extracted)

    query = f"###Prompt {prompts[3]}"
    llm_response = qa(query)
    job_role_predicted = llm_response['result']
    print("Job Role predicted ",job_role_predicted)

    query = f"###Prompt {prompts[4].format(skills_extracted, 15)}"
    llm_response = qa(query)
    screening_questions = llm_response['result']
    print("Screening Questions: ", screening_questions)

    query = f"###Prompt {prompts[5].format(60)}"
    llm_response = qa(query)
    summary = llm_response['result']
    print("summary: ", summary)


    # More evaluation steps here:
    if isEligible.lower() == 'yes':
        isEligible = 1
    else:
        isEligible = 0

    
    # Pushing into Database
    conn = sqlite3.connect(DATABASE_path)
    cur = conn.cursor()
    
    cur.execute("""INSERT INTO applicants (
                    name, 
                    contact_no, 
                    email,
                    projectLinks, 
                    job_role, 
                    resume_file, 
                    skills,
                    isEligible
                    ) VALUES (?,?,?,?,?,?,?,?)""",(name, contact_no, email, projectLinks, job_role, resume_file, skills, isEligible))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Congratulations You have applied for the following role successfully!!"})

if __name__ == "__main__":
    app.run(debug=True)





# Flask Application Building:
