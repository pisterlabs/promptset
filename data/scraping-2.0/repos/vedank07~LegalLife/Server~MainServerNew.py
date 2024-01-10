from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import openai
import fitz

from Code.DataBaseSerch import db_search
from Code.Chromadb import initialize_collection, add_to_collection, process_files, semantic_search
from Code.AskGpt import ask, summarize_text, summarize_big_text, set_openai_api_key


directory_path = 'DataStore Location'
collection, file_list = initialize_collection(directory_path)

if collection and file_list:
    process_files(directory_path, collection, file_list)

set_openai_api_key("Enter Your api Key") #Enter your api key


app = Flask(__name__)
print(99)

data = {
    "message": "Hello, World!"
}

CORS(app) 

@app.route("/api", methods = ["POST"])
def api():

    ##if request.method == "POST":
    data = request.get_json()
    prompt = data.get("Message", "")
    response = ask(prompt)
    response = {"Response": response}
    print(response)

    return jsonify(response), 201


@app.route("/file", methods=["POST"])
def get_file_text():
    # Get the JSON data from the POST request
    data = request.get_json()

    # Extract the 'location' field from the JSON data
    location = data.get("location", "")

    try:
        # Open and extract text content from the PDF file based on the provided location
        if location.lower().endswith('.pdf'):
            # If the file has a .pdf extension, extract text from the PDF file
            doc = fitz.open(location)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            doc.close()
        elif location.lower().endswith('.txt'):
            # If the file has a .txt extension, read text content from the text file
            with open(location, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
        ans = summarize_text(text)


        return jsonify({"Response": ans})
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    
@app.route("/search", methods = ["POST"])
def search():

    ##if request.method == "POST":
    data = request.get_json()
    prompt = data.get("Message", "")
    response = semantic_search(prompt)
    print(response)
    response = {"Response": response}

    return jsonify(response), 201

@app.route("/searchDb", methods = ["POST"])
def searchDb():

    ##if request.method == "POST":
    data = request.get_json()
    prompt = data.get("Message", "")
    response = db_search(prompt)
    print(response)
    response = {"Response": response}

    return jsonify(response), 201

@app.route("/")
def home():
    return "Home"

if __name__ == "__main__":
    app.run(debug=True)

