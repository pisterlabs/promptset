'''
pip3 install -U yt-dlp

!wget -O - -q  https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl

!yt-dlp -xv --ffmpeg-location ffmpeg-master-latest-linux64-gpl/bin --audio-format mp3  -o download.mp3 -- https://youtu.be/yixIc1Ai6jM prev one 'SGzMElJ11Cc'
# Andrew Huberman's Marc Andreessen: How Risk Taking, Innovation & Artificial Intelligence Transform Human Experience : https://www.youtube.com/watch?v=yixIc1Ai6jM

pip3 install pydub
pip3 install pyannote.audio
'''
import os
from pydub import AudioSegment


t1 = 0 * 1000 # works in milliseconds
t2 = 20 * 60 * 1000

newAudio = AudioSegment.from_mp3("download.mp3")
a = newAudio[t1:t2]
a.export("audio1.mp3", format="mp3") 

# Changed all occurances of .wav to .mp3

audio = AudioSegment.from_mp3("audio1.mp3")
spacermilli = 2000
spacer = AudioSegment.silent(duration=spacermilli)
audio = spacer.append(audio, crossfade=0)

audio.export('audio.mp3', format='mp3')

# Use the entire file for the complete transcription || 
# up the char limit on the GPT model

from pyannote.audio import Pipeline

#HF_AUTH = os.getenv('HF_AUTH')

#pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',use_auth_token=HF_AUTH) # why tf is this not working thenv ?

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0",
                                    use_auth_token="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")

DEMO_FILE = {'uri': 'blabal', 'audio': 'audio.mp3'}
dz = pipeline(DEMO_FILE)


with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))

print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")

# Data pre processing : Cleaning up the data

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

import re
dz = open('diarization.txt').read().splitlines()
dzList = []
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = millisec(start) - spacermilli
  end = millisec(end)  - spacermilli
  lex = not re.findall('SPEAKER_01', string=l)
  dzList.append([start, end, lex])

print(*dzList[:10], sep='\n')

from pydub import AudioSegment
import re 

sounds = spacer
segments = []

dz = open('diarization.txt').read().splitlines()
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = int(millisec(start)) #milliseconds
  end = int(millisec(end))  #milliseconds
  
  segments.append(len(sounds))
  sounds = sounds.append(audio[start:end], crossfade=0)
  sounds = sounds.append(spacer, crossfade=0)

sounds.export("dz.mp3", format="mp3") #Exports to a mp3 file in the current path.

segments[:8]

del   sounds, DEMO_FILE, pipeline, spacer,  audio, dz, a, newAudio


# Whisper AI Transcriptions (offline, no need for api keys)
# pip3 install git+https://github.com/openai/whisper.git 


# !whisper dz.mp3 --language en --model large

# !pip install -U webvtt-py

import webvtt

captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read('dz.vtt')]
print(*captions[:8], sep='\n')

# Matching transcriptions with diarations

# we need this for our HTML file (basicly just some styling)
preS = '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <meta http-equiv="X-UA-Compatible" content="ie=edge">\n    <title>Lexicap</title>\n    <style>\n        body {\n            font-family: sans-serif;\n            font-size: 18px;\n            color: #111;\n            padding: 0 0 1em 0;\n        }\n        .l {\n          color: #050;\n        }\n        .s {\n            display: inline-block;\n        }\n        .e {\n            display: inline-block;\n        }\n        .t {\n            display: inline-block;\n        }\n        #player {\n\t\tposition: sticky;\n\t\ttop: 20px;\n\t\tfloat: right;\n\t}\n    </style>\n  </head>\n  <body>\n    <h2>Marc Andreessen: How Risk Taking, Innovation & Artificial Intelligence Transform Human Experience</h2>\n  <div  id="player"></div>\n    <script>\n      var tag = document.createElement(\'script\');\n      tag.src = "https://www.youtube.com/iframe_api";\n      var firstScriptTag = document.getElementsByTagName(\'script\')[0];\n      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n      var player;\n      function onYouTubeIframeAPIReady() {\n        player = new YT.Player(\'player\', {\n          height: \'210\',\n          width: \'340\',\n          videoId: \'yixIc1Ai6jM\',\n        });\n      }\n      function setCurrentTime(timepoint) {\n        player.seekTo(timepoint);\n   player.playVideo();\n   }\n    </script><br>\n'

# xpreS = '<!DOCTYPE html>\n<html lang="en">\n <head>\n <meta charset="UTF-8">\n <meta name="viewport" content="width=device-width, initial-scale=1.0">\n <meta http-equiv="X-UA-Compatible" content="ie=edge">\n< title>Lexicap</title>\n <style>\n body {\n font-family:sans-serif;\n background-color: #f5f5f5;\n margin: 0;\n padding: 0;\n color: #333;\n }\n header {\n background-color: #007BFF;\n color: #fff;\n text-align: center;\n padding: 1em 0;\n }\n .container {\n max-width: 800px;\n margin: 0 auto;\n padding: 20px;\n background-color: #fff;\n box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);\n border-radius: 10px;\n margin-top: 20px;\n }\n        h1 {\n font-size: 32px;\n margin-bottom: 20px;\n }\n        .video-container {\n position: relative;\n padding-bottom: 56.25%;\n height: 0;\n overflow: hidden;\n }\n .video {\n position: absolute;\n top: 0;\n left: 0;\n width: 100%;\n height: 100%;\n }\n  .timestamps {\n margin-top: 20px;\n }\n .timestamp-link {\n color: #007BFF;\n text-decoration: none;\n margin-right: 10px;\n font-weight: bold;\n }\n        .timestamp-link:hover {\n text-decoration: underline;\n }\n .caption {\n margin-top: 10px;\n font-size: 18px;\n color: #555;\n }\n .l {\n color: #050;\n }\n  .s {\n display: inline-block;\n }\n .e {\n  display: inline-block;\n }\n .t {\n  display: inline-block;\n }\n #player {\n position: sticky;\n top: 20px;\n float: right;\n }\n </style>\n </head>\n <body>\n <h2>Yann LeCun: Dark Matter of Intelligence and Self-Supervised Learning | Lex Fridman Podcast #258</h2>\n <div id="player"></div>\n <script>\n var tag = document.createElement('/script/');\n tag.src = "https://www.youtube.com/iframe_api";\n var firstScriptTag = document.getElementsByTagName('script')[0];\n firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n var player;\n function onYouTubeIframeAPIReady() {\n player = new YT.Player('player', {\n height: '210', width: '340',\n videoId: SGzMElJ11Cc,\n });\n }\n function setCurrentTime(timepoint) {\n player.seekTo(timepoint);\n player.playVideo();\n }\n </script><br>\n </body>\n </html>\n'

postS = '\t</body>\n</html>'

from datetime import timedelta

html = list(preS)

for i in range(len(segments)):
  idx = 0
  for idx in range(len(captions)):
    if captions[idx][0] >= (segments[i] - spacermilli):
      break;
  
  while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
    c = captions[idx]  
    
    start = dzList[i][0] + (c[0] -segments[i])

    if start < 0: 
      start = 0
    idx += 1

    start = start / 1000.0
    startStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(start // 3600), 
                                            (int)(start % 3600 // 60), 
                                            start % 60)
    
    html.append('\t\t\t<div class="c">\n')
    html.append(f'\t\t\t\t<a class="l" href="#{startStr}" id="{startStr}">link</a> |\n')
    html.append(f'\t\t\t\t<div class="s"><a href="javascript:void(0);" onclick=setCurrentTime({int(start)})>{startStr}</a></div>\n')
    html.append(f'\t\t\t\t<div class="t">{"[Andrew]" if dzList[i][2] else "[Marc]"} {c[2]}</div>\n')
    html.append('\t\t\t</div>\n\n')

html.append(postS)
s = "".join(html)

with open("Podcast.html", "w") as text_file:
    text_file.write(s)
print(s)


'''

Explain functionality : Takes in audio file from a url and passes it on to openai to transcribe

v1.2 : Archive files from twitter using the link stuff, sent to a specific repository
'''

'''
v1.3 : Diarization + Transcription works alongside Summarisation (Here's a template as to how its done in summarosation.py)

podcast_transcript = result['text']

# !pip3 install openai
# !pip3 install tiktoken

import openai
from getpass import getpass

openai.api_key = getpass('Enter API Key :')

# We can confirm that the API key works by listing all the OpenAI models

models = openai.Model.list()
for model in models["data"]:
 print (model["root"])

import tiktoken
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
print ("Number of tokens in input prompt ", len(enc.encode (podcast_transcript)))

instructPrompt = """
SUMMARIZE THE MAIN BULLET POINTS
"""

request = instructPrompt + podcast_transcript

chatoutput = openai.chatcompletion.create (model="gpt-3.5-turbo-16k",
                                             messages=[{"role": "system","content": "You are a helpful assistant."},
                                                       {"role": "user","content": request}
                                                      ]
)       


podcastSummary = chatoutput.choices[0].message.content
podcastSummary

'''

'''
v1.9.1 : Pre Alpha - Test out functionality for seamless integration into the rest of the stack
'''

'''
v2.0 : Launch and scaling up || Ready for a production environment || Post grant

Save the files according to the name convention and desired format we have put into place
(for both summarizations [plain text] and transcribings [])

Advanced functionality (using an extension) would parse this data from the summarization .txt files and paste it inside designated fields

'''


'''

Modified HTML : Not sure if transcripts show up or not

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Lexicap</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
            color: #333;
        }

        header {
            background-color: #007BFF;
            color: #fff;
            text-align: center;
            padding: 1em 0;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            margin-top: 20px;
        }

        h1 {
            font-size: 32px;
            margin-bottom: 20px;
        }

        .video-container {
            position: relative;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            height: 0;
            overflow: hidden;
        }

        .video {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }

        .timestamps {
            margin-top: 20px;
        }

        .timestamp-link {
            color: #007BFF;
            text-decoration: none;
            margin-right: 10px;
            font-weight: bold;
        }

        .timestamp-link:hover {
            text-decoration: underline;
        }

        .caption {
            margin-top: 10px;
            font-size: 18px;
            color: #555;
        }

        /* Styling from the 1st script */
        .l {
            color: #050;
        }

        .s {
            display: inline-block;
        }

        .e {
            display: inline-block;
        }

        .t {
            display: inline-block;
        }

        #player {
            position: sticky;
            top: 20px;
            float: right;
        }
    </style>
</head>
<body>
    <h2>Yann LeCun: Dark Matter of Intelligence and Self-Supervised Learning | Lex Fridman Podcast #258</h2>
    <div id="player"></div>
    <script>
        var tag = document.createElement('script');
        tag.src = "https://www.youtube.com/iframe_api";
        var firstScriptTag = document.getElementsByTagName('script')[0];
        firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
        var player;
        function onYouTubeIframeAPIReady() {
            player = new YT.Player('player', {
                height: '210',
                width: '340',
                videoId: 'SGzMElJ11Cc',
            });
        }
        function setCurrentTime(timepoint) {
            player.seekTo(timepoint);
            player.playVideo();
        }
    </script><br>
</body>
</html>

'''