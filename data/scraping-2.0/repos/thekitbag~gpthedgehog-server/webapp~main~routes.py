from webapp.main import bp
from flask import jsonify

import openai
import time, os

from flask import json, request, current_app

import speech_recognition as sr
from pydub import AudioSegment
import tempfile

"""The sending of a question to GPT should be a function or class of its own 
probably imported from a separate file, there is a lot of repitition and logic 
which doesnt belong in the routes file IYAM"""

@bp.route('/audio_question', methods=['POST'])
def audio_question():
    if 'audio' in request.files:
        audio_file = request.files['audio']

        # Convert audio file to PCM WAV format using pydub
        try:
            audio_data = AudioSegment.from_file(audio_file)
            audio_data = audio_data.set_frame_rate(16000).set_channels(1)

            # Save the converted audio data to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                audio_data.export(temp_audio_path, format='wav')

        # Create a SpeechRecognition recognizer instance
            recognizer = sr.Recognizer()
        
        except:
            return "unable to transcribe question"

        # Perform speech recognition on the temporary audio file
        with sr.AudioFile(temp_audio_path) as audio:
            audio_data = recognizer.record(audio)
            try:
                text = recognizer.recognize_google(audio_data)
                openai.api_key = current_app.config['GPT_API_KEY']
                
                system_message = "You are Hedgehog, a personal assistant that gives easy to understand answers \
                                    to difficult questions for people who are less familiar with technology. \
                                    Use analogies wherever possible and avoid jargon."
                
                response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                        {"role": "system", "content": system_message},
                                        {"role": "user", "content": text}
                                    ]
                                )
                
                answer = response["choices"][0]["message"]["content"]
                result = {'question': text, 'answer': answer} 
                return jsonify(result)

            except:
                return "unable to transcribe question"

        os.remove(temp_audio_path)

    else:
        return 'No audio file received'


@bp.route('/ask', methods=['GET'])
def ask():
    openai.api_key = current_app.config['GPT_API_KEY']
    user_message = request.args.get("q")
    
    system_message = "You are Hedgehog, a personal assistant that gives easy to understand answers \
                        to difficult questions for people who are less familiar with technology. \
                        Use analogies wherever possible and avoid jargon."
    
    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ]
                    )
    
    answer = response["choices"][0]["message"]["content"]
    """
    time.sleep(3)
    mock_answer = "Chat GPT (Generative Pretrained Transformer) is a machine learning model that's trained to generate \
                    human-like responses to text-based conversations. Think of it like a very smart friend who can listen \
                        to what you say and respond in a way that makes sense based on the context of the conversation. It's \
                            kind of like having a robot that can chat with you!"
    
    """
    question = user_message
    result = {'question': question, 'answer': answer} 
    return jsonify(result)

@bp.route('/follow_up', methods=['POST'])
def follow_up():
    r = json.loads(request.data.decode('utf-8'))

    print(r)

    openai.api_key = current_app.config['GPT_API_KEY']
    
    system_message = "You are Hedgehog, a personal assistant that gives easy to understand answers \
                        to difficult questions for people who are less familiar with technology. \
                        Use analogies wherever possible and avoid jargon."
    
    messages = [{"role": "system", "content": system_message}]

    for i in r['previousQuestions']:
        messages.append({"role": "user", "content": i["q"]})
        messages.append({"role": "assistant", "content": i["a"]})
    
    messages.append({"role": "user", "content": r['followUp']})
    
    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages)

    answer = response["choices"][0]["message"]["content"]
    """
    time.sleep(3)
    mock_answer = "Number two is the second counting number, which comes after one and before three. In terms of ranking, \
                number two is typically next in line after the best or most important thing, like the second-place winner in \
                a competition or the second choice on a list. So you can think of it like the silver medal in the Olympics: it's \
                not as good as first place, but it's still an impressive achievement."
    """
    result = {'question': r['followUp'], 'answer': answer}

    return jsonify(result)