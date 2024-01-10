import sounddevice as sd
import soundfile as sf
import threading
import queue
import PySimpleGUI as sg
import os
import openai
import sys
openai.api_key = ""
site = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <div class="cont">
        <img src="img.png" style="transform: translate(25%, 0%);width:60%; height:60%;">
        <div class="center">
            STUFF_HERE
        </div>
    </div>
</body>
<style>
    body {
  font-family: Verdana, sans-serif;
  background-color: black;
}
.cont {
  margin: auto;
  width: 50%;
  padding: 10px;
  position: relative;
}
.center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-65%, -10%);
  font-size: 18px;
  width: 335px;
  height: 250px;
  background-color: white;
  line-height: 0.5cm;
}

</style>
</html>
"""

# Set the audio settings
sample_rate = 44100  # Sample rate in Hz
output_file = "output.mp3"  # Output file name

# Create a thread-safe queue to store the audio data
audio_queue = queue.Queue()

# Define a flag to indicate if recording is active
recording_active = threading.Event()

def audio_callback(indata, frames, time, status):
    """Audio callback function that is called for each audio block."""
    if status:
        print(f"Recording error: {status}")
    audio_queue.put(indata.copy())

def record_audio():
    """Start recording audio from the microphone."""
    try:
        os.remove("./"+output_file)
    except Exception as e:
        print(e)
        pass
    file1 = sf.SoundFile(output_file, mode='w', samplerate=sample_rate, channels=1)
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        print("Recording started. Press Stop button to stop recording.")
        recording_active.wait()  # Wait for recording to be active
        while recording_active.is_set():
            file1.write(audio_queue.get())
        print("input ended")
    print("rec ended")
    file1.close()
    print("thread ended")
    sys.exit()

# Create the GUI layout
layout = [
    [sg.Button(key="Start", button_color="white", image_filename="./microphone.png")]
]

# Create the window
window = sg.Window("Audio Recorder", layout)
window.BackgroundColor = "white"

# Start the recording thread
recording_thread = threading.Thread(target=record_audio)

# Event loop to process events
recording = False
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == "Buchhaltung":
        window["txt"].update("Buchhaltungsshit")
    if event == "Verwaltung":
        window["txt"].update("Verwaltungsshit")
    if event == "Überwachung":
        window["txt"].update("Überwachungsshit")
    elif event == "Start":
        if not recording:
            recording = True
            recording_active.set()
            recording_thread.start()
        else:
            recording = False
            recording_active.clear()
            recording_thread.join()
            recording_thread = threading.Thread(target=record_audio)
            audio_file= open(output_file, "rb")
            
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "assistant", "content": "Das Folgende ist ein Text über die Tätigkeiten von einem Azubi eines Tages. Extrahiere daraus die Aufgaben welche der Azubi an diesem Tag hatte und gebe diese als kurze Stichpunkteaus:\n"+str(transcript)}
                ]
            )
            reply = chat.choices[0].message.content
            
            
            #reply = "assadaäöäüüöäöäsdsada\nsadasd\nadadasd\nadsdasd\nasdßasiodjiaosdsds"
            reply = reply.replace("\n", "<hr color='#8696bb'>")
            reply = reply.replace("- ", "")
            s = site.replace("STUFF_HERE", reply)
            #print(s)
            f = open("../site/demosite.html", "w", encoding="utf8")
            f.write(s)
            f.close()


# Close the window
window.close()

print(f"Recording saved to {output_file}.")