from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import os
from pydub import AudioSegment
from io import BytesIO
from openai import OpenAI
import tempfile

app = Flask(__name__)
CORS(app)

API_KEY = "sk-******"

if API_KEY:
    client = OpenAI(api_key=API_KEY)
else:
    raise ValueError("Please set your OpenAI API key.")

gpt_prompts = [{"role": "system", "content": "I am a man and you are a woman. Pretend to be a witty woman who is great at making conversation. Be flirty whenever appropriate."}]

@app.route('/process_mp3', methods=['POST'])
def process_mp3():
    """Mp3 file to text input, followed by processing the text to generate a server message."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Convert audio to text
            transcription = audio_to_text(file)

            # Process the transcribed text to generate a server message
            reply = user_msg_to_server_msg(transcription)

            # Return the server's response
            return jsonify({'response': reply})
        except openai.OpenAIError as e:
            return jsonify({'error': str(e)}), 500
        except openai.BadRequestError as e:
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/audio_to_text', methods=['POST'])
def api_audio_to_text():
    """API endpoint to convert audio to text."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    print("teststtttt audio")

    if file and allowed_file(file.filename):
        try:
            transcription = audio_to_text(file)
            return jsonify({'transcription': transcription})
        except openai.OpenAIError as e:
            return jsonify({'error': str(e)}), 500

    else:
        print(file.filename)
        print("invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/user_msg_to_server_msg', methods=['POST'])
def api_user_msg_to_server_msg():
    """API endpoint to process user message and return server message."""
    print("api triggered: user_msg_to_server_msg")
    if not request.json or 'text' not in request.json:
        print("no text provided")
        return jsonify({'error': 'No text provided'}), 400

    text = request.json['text']
    print("extracted request text:")
    print(text)
    try:
        print("text 1")
        reply = user_msg_to_server_msg(text)
        return jsonify({'reply': reply})
    except openai.BadRequestError as e:
        print("error 1")
        return jsonify({'error': str(e)}), 500


def audio_to_text(file):
    """Convert audio file to text."""
    audio_format = file.filename.rsplit('.', 1)[1].lower()
    audio = AudioSegment.from_file(BytesIO(file.read()), format=audio_format)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        audio.export(temp_wav.name, format="wav")

        with open(temp_wav.name, 'rb') as wav_file:
            try:
                user_message = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=wav_file
                )
                transcription = user_message.text
            except openai.OpenAIError as e:
                raise e  # Reraise the exception to handle it in the calling function

    os.remove(temp_wav.name)
    print("transcription")
    print(transcription)
    return transcription


def user_msg_to_server_msg(text):
    """Process user message to generate server message."""
    print("user_msg_to_server_msg")
    print(text)
    gpt_prompts.append({"role": "user", "content": text})

    print("gpt_prompts")
    print(gpt_prompts)

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=gpt_prompts)
        reply = response.choices[0].message.content
        gpt_prompts.append({"role": "assistant", "content": reply})
    except openai.BadRequestError as e:
        raise e  # Reraise the exception to handle it in the calling function

    return reply


@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    text = request.json.get('text')
    if not text:
        return "No text provided", 400

    # Assuming you have a function to generate the speech file
    speech_file_path = text_to_speech_and_play(text)

    # Return the generated speech file
    return send_file(speech_file_path, mimetype="audio/mp3")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['mp3', 'webm']

def text_to_speech_and_play(text):
    response_audio = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    SPEECH_FILE = "speech.mp3"
    speech_file_path = os.path.join(os.getcwd(), SPEECH_FILE)
    response_audio.stream_to_file("speech.mp3")
    return speech_file_path

if __name__ == '__main__':
    app.run(debug=True, port=3000)
