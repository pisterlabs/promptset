from flask import Flask, request, jsonify
# from flask_cors import CORS
import speech_recognition as sr
import openai
import os
# from flask_socketio import SocketIO, emit, join_room


app = Flask(__name__)
#CORS(app)

recognizer = sr.Recognizer()

openai.api_key = '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' #api key updated!!! fiunaly

def process_speech(audio_file):
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        speech_text = recognizer.recognize_google(audio_data)
        
        # gpt api comes into play here
        prompt = f"Debate feedback: {speech_text}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1
        )
        feedback = response.choices[0].text.strip()

        return feedback

    except Exception as e:
        return str(e)

@app.route('/debate', methods=['POST'])
def debate_route():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio source.'}), 400

    try:
        audio_file = request.files['audio']
        feedback = process_speech(audio_file)
        return jsonify({'feedback': feedback})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    # if 'audio' not in request.files:
    #     return jsonify({'error': 'No audio source.'}), 400
    #
    # try:
    #     audio_file = request.files['audio']
    #     feedback = process_speech(audio_file)
    #     return jsonify({'feedback': feedback})

    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    # socketio.run(app, host="0.0.0.0", port=9000)
