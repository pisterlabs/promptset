'''from flask import Flask, request, jsonify
import easyocr
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import openai
openai.api_key = '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'

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}}, supports_credentials=True)
#CORS(app, resources={r"/upload": {"origins": "http://localhost:3001"}})
#CORS(app)
# Initialize the reader
reader = easyocr.Reader(['en']) 

@app.route('/upload', methods=['POST'])
def upload_file():
    #######################IMG_PARSING####################################
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('./src/components', filename)
        file.save(filepath)
        result = reader.readtext(filepath, detail=0)
        os.remove(filepath)  # Remove the file after processing
    ########################GPT_TEXT PARSING##############################
        try:
            response = openai.Completion.create(
            engine="text-davinci-003",
            prompt = f"Given the following text, please extract the 'Question', 'YourAnswer', and 'CorrectAnswer' in json format.\n\nText: \"{result}\"\n\n",
            max_tokens=300
            )
            result = response.choices[0].text.strip()
        except Exception as e:
            result = f"An error occurred: {e}"

        return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
'''
