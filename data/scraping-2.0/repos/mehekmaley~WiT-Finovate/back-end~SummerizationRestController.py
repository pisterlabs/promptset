# using flask_restful
import os
from flask import Flask, jsonify, flash, request, redirect, url_for
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
import numpy as np
import fitz
import openai
from fpdf import FPDF
import json
from azure.storage.blob import BlobServiceClient
from SummarizationService import uploadToBlobStorage, pdfToTextUtil, textToPDF,insertUserMetadata
# creating the flask app
app = Flask(__name__)
cors = CORS(app)

# creating an API object
api = Api(app)
UPLOAD_FOLDER = 'E:\gpt3-summarization\FileDB'
ALLOWED_EXTENSIONS = set(['pdf'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

paperContent = ''
summary = []


# creating an API object
api = Api(app)
UPLOAD_FOLDER = 'E:\gpt3-summarization\FileDB'
ALLOWED_EXTENSIONS = set(['pdf'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

paperContent = ''
summary = []


# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class SummarizationController(Resource):
  
    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):
  
        return jsonify({'message': 'hello world'})
  
    # Corresponds to POST request
    @cross_origin()
    @app.route('/upload', methods=['POST'])
    def upload_file():
        if request.method == 'POST':
            # check if the post request has the file part
            file = request.files['file']
            # if user does not select f ile, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return jsonify({'msg': 'no file received'})
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                completeSummary = pdfToTextUtil(filename)
                print('ans',completeSummary)
                return jsonify({'summaryOfFile': summary})
        return jsonify({'msg': 'response from server'})
    
    @cross_origin()
    @app.route('/saveHistory', methods=['POST'])
    def save_history():
        if request.method == 'POST':
            print("helloworld")
            print("data : ", request.get_json())
            #y = json.loads(request.get_json())
            filename = request.get_json()["fileName"];
            summary = request.get_json()["summary"];
            print("summary : ", summary)
            print("filename : ", filename)
            textToPDF(filename,summary)
            uploadToBlobStorage(filename,filename)
            insertUserMetadata()
        return jsonify({'msg': 'response from server'})

    def allowed_file(filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

api.add_resource(SummarizationController, '/');

# driver function
if __name__ == '__main__':
  
    app.run(debug = True)
