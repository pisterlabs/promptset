from flask import Blueprint, request, render_template, jsonify
import speech_recognition as sr
import os, io, uuid
from textblob import TextBlob
from gtts import gTTS
import openai
openai.api_key = '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' #os.environ.get('OPENAI_API_KEY')

opinions_plus = Blueprint('opinions_plus',__name__, template_folder='templates', static_folder='static')

log = [{"ini": "init log", "answer":'dee84b0b-3a7b-406c-a06c-80aa7aef7d31gen.wav'}]

@opinions_plus.post('/process_audio/')
def process_audio():
    try:
        # Read audio file from request
        data = request.files['audio'].read()

        # Generate a unique filename for the input file
        uuid_name = str(uuid.uuid4())
        filename_in = uuid_name + 'in.wav'

        # Save audio file to disk
        with open(filename_in, 'wb') as f:
            f.write(data)

        # Generate a unique filename for the output file
        filename_out = uuid_name + 'out.wav'

        # Convert Opus codec to WAV format using FFmpeg
        os.system(f'ffmpeg -i {filename_in} -vn -acodec pcm_s16le -ac 1 -ar 16000 {filename_out}')


        # Retrieve audio file from disk and transcribe it
        r = sr.Recognizer()
        with sr.AudioFile(filename_out) as source:
            audio_data = r.record(source)
            transcript = r.recognize_google(audio_data, language='en-EN')

        # Perform sentiment analysis on the transcript
        blob = TextBlob(transcript)
        sentiment = blob.sentiment

        # Return transcription and sentiment as dictionary
        response = {'transcript': transcript, 'sentiment': {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}}
        log.append(response)

        os.remove(filename_in)
        os.remove(filename_out)

        return jsonify(response)

    except Exception as e:
        os.remove(filename_in)
        os.remove(filename_out)
        log.append({"error": f"Error while reading file: {e}"})
        return jsonify({'error': f'Error while processing audio: {e}'}), 400


@opinions_plus.route('/opinions_ai/')
def opinions_gtp():
    return render_template('opinions_ai.html')

@opinions_plus.get("/say_something/")
def api_get_stores():
    return {"methods":log}

@opinions_plus.route('/ask_ai/')
def talk_ai():
    return render_template('ask_ai.html')


@opinions_plus.post('/process_answer/')
def robot_answer():
    try:
        # Read audio file from request
        data = request.files['audio'].read()

        # Generate a unique filename for the input file
        BASE_PATH = '/home/3karopolus/mysite/opinions_plus/static/audio/'
        uuid_name =  str(uuid.uuid4())
        filename_in = BASE_PATH + uuid_name + 'in.wav'

        # Save audio file to disk
        with open(filename_in, 'wb') as f:
            f.write(data)

        # Generate a unique filename for the output file
        filename_out = BASE_PATH + uuid_name + 'out.wav'

        # Convert Opus codec to WAV format using FFmpeg
        os.system(f'ffmpeg -i {filename_in} -vn -acodec pcm_s16le -ac 1 -ar 16000 {filename_out}')


        # Retrieve audio file from disk and transcribe it
        r = sr.Recognizer()
        with sr.AudioFile(filename_out) as source:
            audio_data = r.record(source)
            transcript = r.recognize_google(audio_data, language='es-ES')

        # Use GPT-3 API to generate a response
        response_text = get_gpt_response(transcript)

        # Convert the transcript to audio and save it as a WAV file
        tts = gTTS(text=response_text, lang='es')
        audio_filename = uuid_name + 'gen.wav'
        tts.save(BASE_PATH + audio_filename)

        # Return transcription and sentiment as dictionary
        response = {"uuid":uuid_name,"transcript": transcript, "bot_answer": response_text, "answer": audio_filename}
        log.append(response)

        os.remove(filename_in)
        os.remove(filename_out)

        return jsonify(response)

    except Exception as e:
        os.remove(filename_in)
        os.remove(filename_out)
        log.append({"error": f"Error while reading file: {e}"})
        return jsonify({'error': f'Error while processing audio: {e}'}), 400

@opinions_plus.route('/talk_ai/', methods=['GET'])
def play_answer():
    return render_template('talk_ai.html', log=log)

def get_gpt_response(prompt):
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      temperature=0.5,
      max_tokens=250,
      n=1,
      stop=None,
      presence_penalty=0.6,
      frequency_penalty=0.6,
      best_of=1
    )

    return response.choices[0].text.strip()
