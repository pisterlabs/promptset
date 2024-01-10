from flask import Flask, request, render_template
import os
import subprocess
import datetime

from time import sleep
import openai
from moviepy.video.io.VideoFileClip import VideoFileClip

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():

    def is_video_file(file_path):
        try:
            video = VideoFileClip(file_path)
            return True
        except Exception as e:
            return False

    def video_to_audio(video_file, audio_file):
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(audio_file)


    ## Summarize
    def summarize_text(file_path,apikey):
        # Apply API key
        if apikey=="":
            apikey = open("../api-key.txt", "r").read()
        
        openai.api_key = apikey

        file_path = file_path.replace('\\','/').replace('uploads','audio_transcription')+".txt"

        # Define the text to be summarized
        text = open(f"{file_path}", "r",encoding="utf-8").read()

        limit = 6500
        loops = (len(text)//limit)+1
        message_list = []
        for n in range(loops):
            n +=1
            if (n!=loops) or (n==1):
                t = text[(n-1)*limit:n*limit]
            else:
                t = text[(n-1)*limit:]

            # Define the prompt for summarization
            prompt = f"El siguiente texto es una conversacion, porfavor resumelo y enumera los puntos mas importantes, es importante que no falte ningun tema mencionado:\n\n{t}"

            # Generate a summary
            model_engine = "text-davinci-003"
            completions = openai.Completion.create(engine=model_engine, prompt=prompt, 
                                                   max_tokens=1024, n=1, stop=None, temperature=0.1, 
                                                   frequency_penalty=0, presence_penalty=0)
            message = completions.choices[0].text
            message_list.append(message)


        if len(message_list)==1:
            message_final = "".join(message_list)

        else:
            t2 = "".join(message_list)
            # Define the prompt for summarization
            prompt = f"Porfavor combina estos {len(message_list)} resumenes en uno solo y los puntos enumerados en una sola lista, es importante que no falte ningun tema mencionado:\n\n{t2}"

            # Generate a summary
            model_engine = "text-davinci-003"
            completions = openai.Completion.create(engine=model_engine, prompt=prompt, 
                                                   max_tokens=1024, n=1, stop=None, temperature=0.1, 
                                                   frequency_penalty=0, presence_penalty=0)
            message_final = completions.choices[0].text

        with open(f"results/{file_path.split('.')[0].split('/')[-1]}.txt", 'w') as f:
            f.write(message_final)

        print(message_final)

            
        return message_final


        
    apikey = request.form['apikey']
    file = request.files['file']
    uploads_dir = os.path.join(app.root_path, "uploads")
    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + file.filename
    file_path = os.path.join(uploads_dir, filename)
    file.save(file_path)

    if is_video_file(file_path):
        new_file_path = "".join(file_path.split(".")[:-1])+".mp3"
        video_to_audio(file_path, new_file_path)
        file_path = new_file_path

    command = f"whisper {file_path} --task transcribe --model medium --verbose False --device cuda --output_dir audio_transcription"
    subprocess.run(command, shell=True)


    try:
        message_final = summarize_text(file_path,apikey)
    except Exception as e:
        print('Se intentara de nuevo, hubo un problema de tipo ', e)
        sleep(3)
        try:
            message_final = summarize_text(file_path,apikey)
        except Exception as e:
            print('Otra vez, hubo un problema de tipo ', e)


    message_final = message_final.replace("\n", "<br>")
    return render_template('received.html', message_final=message_final)

if __name__ == '__main__':
    app.run()
