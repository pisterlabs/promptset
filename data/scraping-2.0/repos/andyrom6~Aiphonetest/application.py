import logging
import openai
import os
from flask import Flask, request, jsonify, url_for
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Twilio credentials
twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')

# Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

application = Flask(__name__)

@application.route('/')
def home():
    return 'Hello, World!'
application.run(port=5002)

@application.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('prompt')
    
    if not user_input:
        return jsonify(error='Prompt required'), 400

    try:
        # Use the user's prompt directly without additional structuring
        response = openai.Completion.create(
            engine="text-davinci-003",  # Updated to the latest model for text generation
            prompt=user_input,
            max_tokens=150,  # Adjust based on your needs
            temperature=0.5,  # A balance between randomness and determinism
            n=1  # Get a single completion
        )
        logging.debug(response)
        generated_text = response.choices[0].text.strip()
        return jsonify(output=generated_text, tokens_used=len(generated_text.split()))

    except openai.error.OpenAIError as e:
        logging.exception("OpenAI API request failed")
        return jsonify(error=f"API request failed: {e}"), 500
    except Exception as e:
        logging.exception("An unexpected error occurred")
        return jsonify(error=f"An unexpected error occurred: {e}"), 500

@application.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text')
    if not text:
        return jsonify(error='Text required'), 400

    # Convert text to speech using Twilio
    call = twilio_client.calls.create(
        twiml=f'<Response><Say>{text}</Say></Response>',
        to='+14695775126',
        from_='+14695992660'
    )
    return jsonify(call_sid=call.sid)

@application.route('/voice', methods=['POST'])
def voice():
    response = VoiceResponse()
    # Start recording and transcribe the audio
    response.record(transcribe=True, transcribeCallback=url_for('transcribe', _external=True))
    return str(response)

@application.route('/transcribe', methods=['POST'])
def transcribe():
    transcription_text = request.values.get('TranscriptionText', '')
    if not transcription_text:
        logging.error('No transcription received')
        return '', 400
    # You might want to process or return the transcribed text here
    return jsonify(transcription_text=transcription_text)

if __name__ == '__main__':
    application.run(debug=True)




