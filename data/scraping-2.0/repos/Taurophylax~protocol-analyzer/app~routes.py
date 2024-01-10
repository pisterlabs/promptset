from flask import render_template, request, jsonify, redirect, url_for
import os
import openai
from pydub import AudioSegment
from app import app
from google.cloud import speech
import subprocess 
app.debug = True

# index page - currently blank
@app.route("/")
@app.route("/index")
def index():
    return redirect(url_for('analyzer'))

# analyzer page - where most of the action happens
@app.route("/analyzer", methods=['GET'])
def analyzer():
    status = request.args.get('status')
    outfile = "uploaded_audio.flac" #for status=2
    if status is not None and status == '1': #this means an audio file was uploaded
        #Convert the file to flac -- Requires ffmpeg! -> "sudo apt install ffmpeg"
        infile = os.path.join('uploaded/', request.args.get('file'))
        outfile = os.path.join('uploaded/', "uploaded_audio.flac")
        audio = AudioSegment.from_file(infile)
        #spins up a thread so that we can wait for the conversion to finish
        conversion = subprocess.Popen(
                            ['ffmpeg', '-i', infile, '-y', '-vn', '-acodec', 'flac', '-qscale:a', '0', '-b:a', '48k', outfile])
        conversion.wait() 

        os.remove(infile) #clean up

        # Check if the output file exists
        if os.path.exists(outfile):
            return redirect(url_for('analyzer', status=2)) #success
        else:
            return redirect(url_for('analyzer', status='e2')) #error

    elif status is not None and status == '2':
        return redirect(url_for('execute_transcription', speech_file=outfile))
    return render_template('analyzer.html')

# upload audio file
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        audio_file = request.files['fileupload']
        audio_file.save(os.path.join('uploaded/', audio_file.filename))
        return redirect(url_for('analyzer', status=1, file=audio_file.filename))
    return redirect('analyzer')

# save, convert, and remove audio files
@app.route('/save_audio', methods=['POST'])
def save_audio():
    audio_file = request.files.get('audio')

    if audio_file:
        file_path = os.path.join('uploaded/', audio_file.filename)
        audio_file.save(file_path)
        #Convert webm to flac -- Requires ffmpeg! -> "sudo apt install ffmpeg"
        audio = AudioSegment.from_file(file_path, format="webm")
        flac_path = os.path.join('uploaded/', 'recorded_audio.flac')
        # Save the flac file
        audio.export(flac_path, format="flac", parameters=["-ar", "44100"])
        # Remove the webm file
        os.remove(file_path)
        return jsonify({'success': True}), 200
    else:
        return jsonify({'success': False}), 400

# transcribe audio file
@app.route('/execute_transcription/<speech_file>', methods=['GET'])
def execute_transcription(speech_file):
    speech_file = os.path.join('uploaded/', speech_file)
    client = speech.SpeechClient()
    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    # Google API Calls
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=44100,
        language_code="en-US",
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=30) #30 second timeout

    for result in response.results:
        # The first option, alternatives[0], has the highest confidence
        transcript = format(result.alternatives[0].transcript)
        print("Confidence: {}".format(result.alternatives[0].confidence))  #present confidence score
    os.remove(speech_file) #clean up flac file
    return redirect(url_for('analyzer', status=3, result=transcript))

@app.route('/aianalyze', methods=['GET', 'POST'])
def ai_analyze():
    
    variables = request.args.get('vars')
    inputdata = request.args.get('input')

    sysmsg="Extract the following variables from the following input. The variables are: {}".format(variables) + \
            "Please return the list of variables with their assigned values: for example, author=john and organism=human." + \
            "If a variable is not found, just assign it NA. Add any additional variables that may be important"
    usrmsg=inputdata

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": usrmsg}],
        max_tokens=1024,
        temperature=0,
    )
    response = response.choices[0].message.content
    return jsonify({'result': response})
