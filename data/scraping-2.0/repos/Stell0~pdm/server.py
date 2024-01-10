import os,sys
from typing import List
from flask import Flask, jsonify, flash, request, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from loaderWrapper import LoaderWrapper
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import TextLoader
from db import DB
from queryLLM import QueryLLM
from langchain.docstore.document import Document


UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'epub'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/sources', methods=['GET'])
def get_sources():
    from db import DB
    db = DB()
    
    from sqlalchemy.orm import Session
    import sqlalchemy

    # get all sources already in db
    already_ingested = []
    with Session(db.vectorstore._conn) as session:
        statement = sqlalchemy.text("select distinct(cmetadata->>'source') from public.langchain_pg_embedding;")
        rows = session.execute(statement).fetchall()
        session.close()
        print(rows)
    data = [row[0] for row in rows]
    return jsonify(data)

# delete a source
@app.route('/sources/<string:source>', methods=['DELETE'])
def delete_source(source):
    from db import DB
    db = DB()
    from sqlalchemy.orm import Session
    import sqlalchemy
    with Session(db.vectorstore._conn) as session:
        statement = sqlalchemy.text("delete from public.langchain_pg_embedding where cmetadata->>'source' = '"+source+"';")
        session.execute(statement)
        session.commit()
        session.close()
    return jsonify({'message': 'Source deleted'})

# upload a file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            # load file
            loader = LoaderWrapper(path=path, type="file")
            documents = loader.load()
            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
            
            # print documents to standard output for debugging
            for index,document in enumerate(documents):
                print("Document "+str(index))
                print(document.page_content)
                print(document.metadata)
                print("-"*80)

            from db import DB
            db = DB()
            ids = db.vectorstore.add_documents(documents)
            return redirect("/sources",200)
        else:
            return redirect("/sources",400)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file accept=".txt,.pdf,.epub">
      <input type=submit value=Upload>
    </form>
    '''

# Ask a question to your data
@app.route('/ask', methods=['POST'])
def ask():
	question = request.json.get('question')
	history_array = request.json.get('history', '')
	filter = request.json.get('filter', '')
	try:
		history = []
		for [answer, question] in history_array:
			history.append((answer, question))
	except:
		history = ""
	oracle = QueryLLM()
    #print(oracle.ask(question, history))
    #return jsonify({'message': 'Question answered'})
	return jsonify(oracle.ask(question, history, filter))

# get source
@app.route('/sources/<string:source>', methods=['GET'])
def get_source(source):
    #from urllib.parse import unquote
    #source = unquote(source)
    from db import DB
    db = DB()
    from sqlalchemy.orm import Session
    import sqlalchemy
    with Session(db.vectorstore._conn) as session:
        statement = sqlalchemy.text("select * from public.langchain_pg_embedding where cmetadata->>'source' = '"+source+"';")
        rows = session.execute(statement).fetchall()
        session.close()
        print(rows)
    data = []
    for row in rows:
        data.append({"id":row[0], "metadata":row[2]})
    return jsonify(data)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
