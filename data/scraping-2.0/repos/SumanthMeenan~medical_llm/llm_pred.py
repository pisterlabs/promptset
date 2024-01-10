from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)

# Set your OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    try:
        # Get the uploaded text file
        file = request.files['file']
        patient_history = file.read().decode("utf-8")

        # Call the OpenAI API to generate answers
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=patient_history,
            max_tokens=150  # Limit the response length
        )

        # Extract the generated answer from the API response
        answer = response.choices[0].text.strip()

        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)