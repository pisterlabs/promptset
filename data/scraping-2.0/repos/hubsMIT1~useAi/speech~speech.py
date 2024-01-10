# # import flask from Flask , render_template,url_for,redirect

# import openai
#
# import pyttsx3
#
# import speech_recognition as sr
#
# import time
#
#
#
# openai.api_key = "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"


# engine = pyttsx3.init()

# def transcribe_audio_to_text(filename):

#     recognizer = sr.Recognizer()

#     with sr.AudioFile(filename) as source:

#         audio = recognizer.record(source)

#     try:

#         return recognizer.recognize_google(audio)

#     except Exception as e:

#         print("Skipping unknown error:", e)

#         return ""

# def generate_response(prompt):

#     response = openai.Completion.create(

#         engine="text-davinci-002",

#         prompt=prompt,

#         max_tokens=4000,

#         n=1,

#         stop=None,

#         temperature=0.5,

#     )

#     return response.choices[0].text

# def speak_text(text):

#     engine.say(text)

#     engine.runAndWait()

# def main():
#     while True:
        
#         print("Say 'you' to start recording your question...")
#         with sr.Microphone() as source:
#             recognizer = sr.Recognizer()
#             audio = recognizer.listen(source)
#             try:
#                 transcription = recognizer.recognize_google(audio)
#                 if transcription.lower() == "you":
#                     # Record audio
#                     filename = "input.wav"
#                     print("Say your question...")
#                     with sr.Microphone() as source:
#                         recognizer = sr.Recognizer()
#                         source.pause_threshold = 1
#                         audio = recognizer.listen(source, phrase_time_limit=10, timeout=None)

#                         with open(filename, "wb") as f:
#                             f.write(audio.get_wav_data())

                    
#                     text = transcribe_audio_to_text(filename)
#                     if text:
#                         print(f"You said: {text}")
                        
#                         response = generate_response(text)
#                         print(f"GPT-3 says: {response}")
                        
#                         speak_text(response)
#                 else:
#                     print("Please try again and say 'Genius' to start recording...")
#             except Exception as e:
#                 print(f"An error occurred: {e}")
                
# if __name__ == "__main__":
#     main()

from flask import Flask, request, make_response
from flask_cors import CORS,cross_origin
app = Flask(__name__)
# CORS(app)
@app.route('/audio', methods=['POST'])
@cross_origin(origin="*", headers=["Content-Type"])
def audio():
    # response = make_response('Success')
    # response.headers['Access-Control-Allow-Origin'] = '*'
    # response.headers['Content-Type'] = 'text/plain'
    audio_data = request.data
    # Process audio data
    print("success")
    return 'Audio received'

if __name__ == '__main__':
    app.run(debug=True)
