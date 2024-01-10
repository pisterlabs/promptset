from flask import Flask, request, render_template, jsonify
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_name = "google/t5-large-ssm-nq"
#model_name = "google/flan-t5-xxl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
file_text=''


def extract_text_from_pdf(pdf_file):
    text = ''
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()

    return text

def handle_file_upload():
    try:
        pdf_file = request.files['pdffiles']

        # Check if the file is not None
        if pdf_file is None:
            return jsonify({'error': 'No file uploaded.'}), 400

        # Check file content type
        if pdf_file.content_type != 'application/pdf':
            return jsonify({'error': 'Invalid file format. Please upload a PDF file.'}), 400

        # Extract text from PDF
        try:
            file_text = extract_text_from_pdf(pdf_file)
            return jsonify({'result': "File Upload Success!"})
        except Exception as e:
            print("Error extracting text:", e)
            return jsonify({'error': str(e)}), 400

    except Exception as e:
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred.'}), 500


# Function to process user queries and get model responses
def get_model_response(query, pdf_text):
    input_text = "question: {} context: {}".format(query, pdf_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate model response
    with torch.no_grad():
        model_output = model.generate(inputs)

    # Decode the model's response
    model_response = tokenizer.decode(model_output[0], skip_special_tokens=True)
    return model_response

def generate_message():
    try:
        # Handle text input
        text = request.form["msg"]
        model_response = get_model_response(text, file_text)
        return jsonify({'result': model_response })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


app  = Flask(__name__)
@app.route('/')
def home():
    return  render_template("index.html")


@app.route('/get', methods=['POST', 'POST1'])
def chatbot_response():
    if request.method == 'POST':
        return handle_file_upload()
    elif request.method == 'POST1':
        return generate_message()
    else:
        return jsonify({'result': "Bot not found"})



@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__=="__main__":
    app.run(debug=True)