

from flask import Flask, request, jsonify, send_from_directory
import PyPDF2
import io
import os
import openai

app = Flask(__name__)

# Set debug mode (set to True for testing without OpenAI calls)
DEBUG_MODE = False 


openai.api_key = "your api key" 
openai.api_base = "your endpoint url"

openai.api_type = 'azure'
openai.api_version = '2023-09-15-preview' # This API version or later is required to access fine-tuning for turbo/babbage-002/davinci-002

@app.route('/')
def index():
    # Serve the HTML file
    return send_from_directory('', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        pdfReader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        responses = []
        combined_response = ""

        for page_number, page in enumerate(pdfReader.pages):
            text = page.extract_text()
            if DEBUG_MODE:
                # In debug mode, return the first 10 characters of each page
                response_text = text[:10]
            else:
                # Constructing the prompt for the model
                prompt = f"Review this legal document excerpt for any elements that seem legally unfair under US law \"{text}\""
                
                response = openai.ChatCompletion.create(
                    engine="deploy_V1", # Replace with your custom model name
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant reviewing legal documents."},
                        {"role": "user", "content": prompt}
                    ]
                )
                # Extract response from OpenAI
                response_text = response['choices'][0]['message']['content']

            # Collecting responses for each page
            combined_response += f"Page {page_number + 1}: {response_text}\n\n"

        return jsonify({"text": combined_response})

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    app.run(debug=True)