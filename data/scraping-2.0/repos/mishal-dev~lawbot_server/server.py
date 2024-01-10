from flask import Flask, request, render_template, redirect, jsonify
# from werkzeug.utils import secure_filename
import requests

import speech_recognition as sr
from gtts import gTTS
import vercel_ai
import os
from pydub import AudioSegment
from googletrans import Translator
from uuid import uuid4
import logging
from flask_cors import CORS
# from langchain import PromptTemplate, LLMChain
# from langchain.llms import GPT4All
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
app = Flask(__name__)
cors = CORS(app)
cors = CORS(app, resources={"/*": {"origins": "*"}})


def get_public_ip():
    try:
        # Make a GET request to httpbin's IP endpoint
        response = requests.get("https://httpbin.org/ip")
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response and extract the IP address
            public_ip = response.json()["origin"]
            return public_ip
        else:
            print(f"Request failed with status code {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return None

public_ip = get_public_ip()
if public_ip:
    print(f"Public IP Address: {public_ip}")
else:
    print("Failed to retrieve public IP address.")

try:
    os.mkdir("uploads")
except:
    print("Folder exists")

path = os.path.abspath("uploads")
user_id = None


# def mygpt(doubt):
#     print("inside mygpt")


# # from langchain.document_loaders import TextLoader

# # loader = TextLoader("/Users/mishalahammed/Downloads/COI_.txt")

#     question_template = """Question: {question}

#     Answer: Let's think step by step."""

#     prompt = PromptTemplate(template=question_template, input_variables=["question"])
#     local_path = (
#         "/Users/mishalahammed/Downloads/nous-hermes-13b.ggmlv3.q4_0.bin"  # replace with your desired local file path
#     )

#     callbacks = [StreamingStdOutCallbackHandler()]

#     llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

#     llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)


#     llm_chain = LLMChain(prompt=prompt, llm=llm,verbose=False)


#     answer=llm_chain.run(doubt)
#     return answer



def audio_convert():
    input_wav_file = os.path.join(path, f"{user_id}.wav")
    output_flac_file = os.path.join(path, f"{user_id}.flac")

    audio = AudioSegment.from_file(input_wav_file, format="webm")
    audio.export(output_flac_file, format="flac")

    print(f"Conversion complete. FLAC file saved as {output_flac_file}")
    return output_flac_file

def chatgpt():
    print("chatgpt is here to help")
    flac_audio = audio_convert()
    recognizer = sr.Recognizer()
    song = sr.AudioFile(flac_audio)

    print("song")
    with song as source:
        print("song is being analyzed")
        song = recognizer.record(source)
    print("song is analyzed")
    song_text = recognizer.recognize_google(song, language=f"{user_language}-IN")

    print(song_text)
    translator = Translator()
    translation = translator.translate(song_text)
    prompt = "I'm an indian citizen and how does indian law help in the following " + str(translation.text) + " please mention the article numbers as well"
    print(prompt)
    #answer=mygpt(prompt)
    client = vercel_ai.Client()
    def model():
        try:
            answer = ""
            for chunk in client.generate("openai:gpt-3.5-turbo", prompt):
                answer += chunk
        except Exception as e:
            model()
        return answer
    answer=model()


    answer=translator.translate(text=answer,src='auto',dest=f'{user_language}').text


    os.remove(os.path.join(path, f"{user_id}.wav"))
    os.remove(os.path.join(path, f"{user_id}.flac"))
    return answer

@app.route("/", methods=["GET", "POST"])
def homepage():
    global user_id
    user_id = str(uuid4())
    print(user_id)
    return render_template("index.html")
@app.route('/data', methods=['GET'])
def get_data():
    data = {"message": "Hello from Flask!"}
    return jsonify(data)
@app.route("/upload", methods=["POST"])
def upload_audio():
    global user_id
    print("audio written")

    try:    
        uploaded_file = request.files['audio']
        global user_language
        user_language=request.form["language"]
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join(path, f"{user_id}.wav"))
            print("Audio saved")
            answer=chatgpt()
            data={"message":answer}
            return jsonify(data)
    except Exception as e:
        print(e)
        data={"message":"error"}
        return jsonify(data)

# @app.route("/lawbot", methods=["GET", "POST"])
# def final_page():
#     return render_template("final.html", long_paragraph=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8000",debug=True)
