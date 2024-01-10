import keyboard
import os
import re
import time
import openai
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from playsound import playsound

# loads env variables file
load_dotenv()

### AUTH KEYS ###

AZURE_SPEECH_KEY = os.getenv("AZURE") #AZURE
OAI_API_KEY = os.getenv("YOUR_API_KEY") #OPENAI
openai.api_key=OAI_API_KEY #OPEN AI INIT

### AZURE ###

# configs tts
speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region="eastus")

## STT LANGUAGES ##

speech_config.speech_recognition_language="en-US"

#speech_config.speech_recognition_language="es-US"
#speech_config.speech_recognition_language="es-MX"
#speech_config.speech_recognition_language="es-PR"
#speech_config.speech_recognition_language="es-DO"
#speech_config.speech_recognition_language="es-SV"
#speech_config.speech_recognition_language="es-CU"

#speech_config.speech_recognition_language="yue-CN"
#speech_config.speech_recognition_language="zh-CN"

#speech_config.speech_recognition_language="vi-VN"

#speech_config.speech_recognition_language="ru-RU"

#speech_config.speech_recognition_language="ar-EG"
#speech_config.speech_recognition_language="ar-SY"
#speech_config.speech_recognition_language="ar-MA"

#speech_config.speech_recognition_language="fr-FR"

#speech_config.speech_recognition_language="km-KH"

#speech_config.speech_recognition_language="it-IT"

#speech_config.speech_recognition_language="fil-PH"

#speech_config.speech_recognition_language="ja-JP"

## TTS LANGUAGES ##
# other than Aria, style compatible (-empathetic) with Davis, Guy, Jane, Jason, Jenny, Nancy, Tony

# ENGLISH #
#speech_config.speech_synthesis_voice_name='en-US-NancyNeural'
#speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
#speech_config.speech_synthesis_voice_name='en-US-DavisNeural'
#speech_config.speech_synthesis_voice_name='en-US-GuyNeural'
#speech_config.speech_synthesis_voice_name='en-US-JaneNeural'
#speech_config.speech_synthesis_voice_name='en-US-JasonNeural'
#speech_config.speech_synthesis_voice_name='en-US-SaraNeural'
#speech_config.speech_synthesis_voice_name='en-US-TonyNeural'
speech_config.speech_synthesis_voice_name='en-US-AriaNeural'

# CHINESE #
#speech_config.speech_synthesis_voice_name='zh-CN-XiaohanNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-XiaomoNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-XiaoruiNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-XiaoxiaoNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-XiaoxuanNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-XiaoyiNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-XiaozhenNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-YunfengNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-YunxiNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-YunyeNeural'
#speech_config.speech_synthesis_voice_name='zh-CN-YunzeNeural'

# SPANISH #
#speech_config.speech_synthesis_voice_name='es-US-PalomaNeural' # united states
#speech_config.speech_synthesis_voice_name='es-MX-CarlotaNeural' # mexican
#speech_config.speech_synthesis_voice_name='es-PR-KarinaNeural' # puerto rican
#speech_config.speech_synthesis_voice_name='es-DO-RamonaNeural' # dominican
#speech_config.speech_synthesis_voice_name='es-SV-LorenaNeural' # salvadorean
#speech_config.speech_synthesis_voice_name='es-CU-BelkysNeural' # cuban

# CHINESE #
#speech_config.speech_synthesis_voice_name='yue-CN-XiaoMinNeural' # cantonese
#speech_config.speech_synthesis_voice_name='zh-CN-XiaochenNeural' # mandarin

# VIETNAMESE #
#speech_config.speech_synthesis_voice_name='vi-VN-HoaiMyNeural'

# RUSSIAN #
#speech_config.speech_synthesis_voice_name='ru-RU-DariyaNeural'

# ARABIC #
#speech_config.speech_synthesis_voice_name='ar-EG-SalmaNeural' # egyptian
#speech_config.speech_synthesis_voice_name='ar-SY-AmanyNeural' # syrian
#speech_config.speech_synthesis_voice_name='ar-MA-MounaNeural' # moroccan

# FRENCH #
#speech_config.speech_synthesis_voice_name='fr-FR-BrigitteNeural'

# KHMER #
#speech_config.speech_synthesis_voice_name='km-KH-SreymomNeural'

# ITALIAN #
#speech_config.speech_synthesis_voice_name='it-IT-ElsaNeural'

# TAGALOG #
#speech_config.speech_synthesis_voice_name='fil-PH-BlessicaNeural'

# JAPANESE #
#speech_config.speech_synthesis_voice_name='ja-JP-MayuNeural'

# sets voice
voice = speech_config.speech_synthesis_voice_name

# sets tts sample rate
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm)

# microphone device stt 
stt_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
# speaker device tts
tts_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# inits stt
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=stt_config)
# inits tts
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=tts_config)

# sets up identifiers for conversation
bot = "User"
user = "Bash"

### SETUP VARIABLES ###
# concats message history for re-insertion with every prompt
context = ""
messages = []
# holds emotional response chosen by GPT-3
style = ""
# counts number of times user silence for input
silenceCount = 0
# counts number of messages in conversation history 
messageCount = 0
# saves last response by the bot
responsePrevious = ""
response = ""

# inputs and reads user prompt
# chooses emotional response from given list of styles
# returns style/emotion
def tone_gpt3(zice, raspuns):
    toneLabel = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Read the following interaction, then pick just one of the emotions for "+bot+" to respond to "+user+" with from this list only: [friendly, empathetic, cheerful, excited, hopeful, unfriendly, angry, shouting, sad, terrified, whispering].\n"+bot+": "+raspuns+"\n"+user+": "+zice+"\n\nEmotion: [", # EN ARIA
        #prompt="Read the following interaction, then pick just one of the emotions for "+bot+" to respond to "+user+" with from this list only: [friendly, empathetic, cheerful, excited, hopeful, unfriendly, angry, shouting, sad, terrified, whispering].\n"+bot+": "+raspuns+"\n"+user+": "+zice+"\n\nEmotion: [", # EN
        #prompt="Read the following interaction, then pick just one of the emotions for "+bot+" to respond to "+user+" with from this list only: [friendly, empathetic, calm, fearful, disgruntled, serious, embarassed, depressed, lyrical, poetry-reading, affectionate, cheerful, excited, hopeful, unfriendly, angry, shouting, sad, terrified, whispering].\n"+bot+": "+raspuns+"\n"+user+": "+zice+"\n\nEmotion: [", # CN
        temperature=1.0,
        max_tokens=12,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=[user+":", bot+":", "Emotion: [", ","],
    )
    return toneLabel

def concatenate_context():
    
    global messages
    global context
    
    if len(messages) == 6:
        messages.pop()
        
    #print(len(messages))
        
    for message in messages:
        context += message

# inputs and reads patient prompt
# responds with given style from TONE_GPT3()
# returns response
def chat_gpt3(zice):
    start_time = time.time()
    reply = openai.Completion.create(
        engine="text-davinci-003",
        prompt= "This is a chat between "+bot+" and "+user+". You are "+bot+". "+bot+" loves to have interesting and random conversations. "+bot+" is also very funny and witty. "+bot+" speaks informally and casually, constantly using filler words such as 'um', 'uh', 'like', 'very', 'really', 'just', 'you know', 'i mean', 'i guess', 'totally', 'literally', 'seriously', and others when speaking."+context+"\n"+user+": "+zice+"\n"+bot+" ["+style+"]:",
        temperature=1.0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=2.0,
        presence_penalty=2.0,
        stop=[user+":", bot+":", "["],
        echo=False,
        stream=True,
    )
    responseTime = time.time() - start_time
    
    # create variables to collect the stream of events
    collected_events = []
    completion_text = ""
    
    print(f"{bot} [{style}]:", end="")

    # iterate through the stream of events
    for event in reply:
        event_time = time.time() - start_time  # calculate the time delay of the event
        collected_events.append(event)  # save the event response
        event_text = event['choices'][0]['text']  # extract the text
        # Encode the string using the utf-8 codec
        encoded_text = event_text.encode('utf-8')
        decoded_text = encoded_text.decode('utf-8')
        completion_text += decoded_text  # append the text
        print(decoded_text, end="")  # print the delay and text
        
    # print response time
    print(f" [{responseTime:.2f} S]\n")
        
    return completion_text

# inputs response SSML from CHAT_GPT()
# streams async synthesis
def tts(ssml):
    global speech_synthesis_result
    #speech_recognizer.stop_speaking_async()
    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()
    #speech_synthesis_result = speech_synthesizer.start_speaking_ssml_async(ssml)
"""
#WHISPER SPEECH TO TEXT IMPLEMENTATION

def stt(model="base", english=False, verbose=False, energy=300, pause=0.5, dynamic=True):
    
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "temp.wav")

    audio_model = whisper.load_model(model)
    
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic

    with sr.Microphone(sample_rate=16000) as source:
            
        # prints status
        print("|||||||||| LISTENING ||||||||||")
        
        audio = r.listen(source)
        data = io.BytesIO(audio.get_wav_data())
        audio_clip = AudioSegment.from_file(data)
        audio_clip.export(save_path, format="wav")

        if english:
            result = audio_model.transcribe(save_path,language='english')
        else:
            result = audio_model.transcribe(save_path)

        if not verbose:
            predicted_text = result["text"]
            print("You said: " + predicted_text)
        else:
            print(result)
    
    return predicted_text
"""
def respond(prompt, response):
    
    global messages
    global silenceCount
    
    responseFormatted = f"{bot}:" + response

    messages.append("\n"+prompt+"\n"+responseFormatted)

    # concats message to memory/history
    concatenate_context()

    # SSML for TTS with response and style
    xmlString = '''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="'''+voice+'''">
    <prosody rate="medium">
    <mstts:express-as style="'''+style+'''" styledegree="1">
    '''+ response +'''
    </mstts:express-as>
    </prosody>
    </voice>
    </speak>'''

    # synthesizes TTS with input SSML
    tts(xmlString)

    # resets silence count to 0
    silenceCount = 0
    
# given input stt
# generates style and response from GPT-3
# synthesizes response tts
def think(inp):
    
    global silenceCount
    global style
    global responsePrevious
    
    # checks if there is verbal input
    if inp != "":
        
        # parses and formats patient input
        prompt = user+": "+inp
        print("\n\n"+prompt)
        
        # gets style tone
        style = ((tone_gpt3(inp, responsePrevious)).choices[0].text).split("]")[0]
        
        # gets GPT text message response completion
        responsePrevious = chat_gpt3(inp)
        
        respond(prompt, responsePrevious)
        
        return
    
    # assumes there is no input
    # checks if has been silent for three rounds
    elif silenceCount == 2:
        
        # imitates silent input
        prompt = user+": ..."
        print("\n\n"+prompt)
        
        # gets style tone
        style = ((tone_gpt3(inp, responsePrevious)).choices[0].text).split("]")[0]
        
        # gets GPT text message response completion
        responsePrevious = chat_gpt3("...")
        
        respond(prompt, responsePrevious)
        
        return
            
    # increases silence count
    silenceCount += 1
    
def listeningAnimation():
    
    listening = "||||||||||"
    
    for character in listening:
        time.sleep(0.001)
        print(character, end="")
        
def recognize():
    
    # gets azure stt
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    #speech_recognizer.start_continuous_recognition_async()
    
    return speech_recognition_result

def listen():
    
    # listens for speech
    while True:

        playsound('start.mp3', False)
        
        listeningAnimation()
        
        speech_recognition_result = recognize()
        
        playsound('stop.mp3', False)

        # gets tts from azure stt
        speech_recognizer.recognized.connect(think(speech_recognition_result.text))

        #message = input(patient + ": ")
        #think(message)
        
def wait_for_key(key):
    
    while True:  # making a loop
        if keyboard.is_pressed(key):  # if key is pressed
            break  # finishing the loop
        
print("\ngpt-s2s\n\nwait for the |||||||||| command and sound cue before speaking.\n\npress the space key to continue...\n")

wait_for_key('space')

listen()