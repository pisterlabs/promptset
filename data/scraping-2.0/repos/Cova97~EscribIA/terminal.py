import os
import openai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Cargar la aplicacion
app = Flask(__name__)
CORS(app)

# Cargar las variables de entorno
load_dotenv()

# Cargar la API key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Funcion para corregir el texto
def correct_spanish(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Correct this to standard Spanish:\n\n{text}",
        temperature=0.5,
        max_tokens=1500,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    corrected_text = response.choices[0].text.strip()
    return corrected_text

# Funcion para calificar el texto
def score_spanish(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"This is a Spanish text:\n\n{text}\n\nThe score is",
        temperature=0.0,
        max_tokens=1500,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    score_text = response.choices[0].text.strip()
    return score_text

# Ruta para corregir el texto
@app.route('/', methods=['POST'])
def correct_text():
    data = request.get_json()
    text = data['text']
    corrected_text = correct_spanish(text)
    score_text = score_spanish(text)
    return jsonify({'corrected_text': corrected_text, 'score_text': score_text})

# main
if __name__ == '__main__':
    app.run(debug=True)