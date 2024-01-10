from backend.SoundExtractor import *
from backend.SoundTranscriptor import *
import os
import openai
#Disable HTTPS verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
openai.api_key = os.getenv("OPENAI_API_KEY")

videoFile = "mp4/input.mp4"
audioFile = "mp4/input.mp3"
textFile  = "mp4/input_transcribed.txt"
summaryFile  = "mp4/input_summary.txt"
soundExtractor = SoundExtractor()
soundTranscriptor = SoundTranscriptor()
#video to sound
soundExtractor.setVideoFile(videoFile)
soundExtractor.setAudioFile(audioFile)
soundExtractor.extractAudio()
# Audio 2 text
print(f"Available models: {soundTranscriptor.fetchAvailableModel()}")
soundTranscriptor.setAudioFile(audioFile)
soundTranscriptor.setTextFile(textFile)
soundTranscriptor.transcribe()
# Resume with open AI
with open(textFile, "r") as file:
    contents = file.read()
    print(contents)
prompt = "Summarize me in the following text:" + contents

model = "text-davinci-002"
completions = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7,
)
message = completions.choices[0].text
print (''' 
Summary
=======
''')
print(message)
with open(summaryFile, "w") as file:
    file.write(message)
