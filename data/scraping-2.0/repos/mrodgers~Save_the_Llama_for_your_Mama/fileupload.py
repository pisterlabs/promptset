from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from flask import jsonify

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os

DATA_PATH = '/home/azureuser/Llama2-hf/data'
DB_FAISS_PATH = "vectorstores/db_faiss/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/azureuser/Llama2-hf/data'
app.config['DATA_PATH'] = '/home/azureuser/Llama2-hf/data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max-limit
app.secret_key = 'supersecretkey'

API_KEY = 'key'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', files=files)


from flask import jsonify
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        api_key = request.form.get('api_key')  # Note the change here
        if not api_key or api_key != API_KEY:
            flash('Invalid API key')
            return redirect(request.url)
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        
        if not files or all(file.filename == '' for file in files):
            flash('No selected file')
            return redirect(request.url)
        
        file_errors = []
        for file in files:
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                file_errors.append(file.filename)
        
        if file_errors:
            flash('Error with files: ' + ', '.join(file_errors))
            flash('Allowed file type is PDF')
            return redirect(request.url)
        
        flash('Files successfully uploaded')
        return redirect(url_for('index'))
    else:
            return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_directory():
    api_key = request.form.get('api_key')  # Note the change here
    if not api_key or api_key != API_KEY:
        flash('Invalid API key')
        return redirect(request.url)
    try:
        print("Data Processing Beginning...")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print("Data Processing Complete.")
        
        return jsonify(success=True, message="Directory Embeddings Processing Complete."), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify(success=False, message=f"An error occurred: {e}"), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
