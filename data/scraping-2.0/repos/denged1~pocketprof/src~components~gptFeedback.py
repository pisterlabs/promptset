from flask import Flask, request, jsonify
import easyocr
import os
import openai
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust the origins as needed

# Set up your OpenAI API key
openai.api_key = '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'

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    if file:
        # Save the file and perform OCR
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)  # Use a temp directory to save the file
        file.save(filepath)
        result = reader.readtext(filepath, detail=0)
        os.remove(filepath)  # Clean up the file after processing
        
        # Now call GPT-3 to parse the text
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Given the following text, please extract the 'Question', 'YourAnswer', and 'CorrectAnswer' in JSON format.\n\nText: \"{result}\"\n\n",
                max_tokens=300
            )
            parsed_result = response.choices[0].text.strip()
        except Exception as e:
            parsed_result = f"An error occurred: {e}"

        return jsonify(result=parsed_result)

@app.route('/generate-feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    question = data['question']
    userAnswer = data['userAnswer']
    correctAnswer = data['correctAnswer']

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Given a Question '{question}'\n My Answer '{userAnswer}'\n And the Correct Answer '{correctAnswer}'\n Can you explain exactly what I did wrong, with step by step explanations?",
            max_tokens=300
        )
        feedback = response.choices[0].text.strip()
        return jsonify(feedback=feedback)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)



'''from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

openai.api_key = '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'

@app.route('/generate-feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    question = data['question']
    answer = data['userAnswer']
    correct_answer = data['correctAnswer']

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Given a Question '{question}'\n My Answer '{answer}'\n And the correct Answer '{correct_answer}'\n Can you explain exactly what I did wrong, with step by step explanations?",
            max_tokens=300
        )
        result = response.choices[0].text.strip()
        return jsonify(feedback=result)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)'''
