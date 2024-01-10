import os
import openai
import elevenlabs as el
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for
from flask_cors import CORS
import json
import tempfile

load_dotenv()

el.set_api_key(os.getenv("ELEVENLABS_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():

    messages = json.loads(request.form.get('messages'))
    user_audio_obj = request.files.get('user_audio')

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file_name = temp_file.name

    with open(temp_file_name, 'wb') as temp_file:
        user_audio_obj.save(temp_file_name)

    with open(temp_file_name, 'rb') as temp_file:
        user_transcript = openai.Audio.transcribe('whisper-1', temp_file, language='es')['text']

    os.remove(temp_file_name)

    print(user_transcript)

    messages.append({
        "role": "user",
        "content": user_transcript
    })

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=messages
    )

    user_text = user_transcript
    assistant_text = response['choices'][0]['message']['content']

    return {
        'status': 0,
        'errorMessage': '',
        'data': {
            'userText': user_text,
            'assistantText': assistant_text
        }
    }

app.run(debug=False, port=5000, host='0.0.0.0')
