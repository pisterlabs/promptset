
# ############################################################################
#
#   home-ai.py
#
#   A ChatGPT based home assistant
#
#   (c) 2023 by Dirk Braner (zapccu) - d.braner@gmx.net
#
# ############################################################################


import configparser as cp
import argparse
import speech_recognition as sr
import boto3
import pyaudio
import pygame
import sys
import os
import openai
import wave
from contextlib import closing
from botocore.exceptions import BotoCoreError, ClientError

VERSION = "1.0.0"

# Configuration
CONFIG = cp.ConfigParser()

# Set default parameters
CONFIG['common'] = {
    'activationWord': 'computer',
    'duration': 3,
    'energyThreshold': -1,
    'sampleRate': 44100,
    'audiofiles': os.path.dirname(os.path.realpath(__file__)) + "/audio"
}
CONFIG['commands'] = {
    'stop': 'stop',
    'mute': 'mute',
    'unmute': 'unmute',
    'terminate': 'terminate'
}
CONFIG['Google'] = {
    'language': 'en-GB'
}
CONFIG['AWS'] = {
    'awsKeyId': 'none',
    'awsKeySecret': 'none',
    'region': 'eu-west-2',
    'pollyVoiceId': 'Brian',
    'language': 'en-GB'
}
CONFIG['OpenAI'] = {
    'openAIKey': 'none',
    'openAILanguage': 'en',
    'openAIModel': 'gpt-3.5-turbo'
}
CONFIG['messages'] = {
    'welcome': 'Hello, I am your personal artificial intelligence. Please say the activation word {activationWword}, if you like to ask me anything.',
    'didNotUnderstand': 'Sorry, I did not understand this',
    'shutdown': 'Shutting down',
    'genericError': 'Something went wrong',
    'muted': 'I am currently inactive. Activate me to ask questions' 
}

# Audio recording parameters
READ_CHUNK       = 4096     # Chunk size for output of audio data >4K
CHANNELS         = 1        # Mono
BYTES_PER_SAMPLE = 2        # Bytes per sample

# Log details level
LOG_LEVEL = 0               # 0 = Print errors only, 1 = Print some more information, 2 = Print debug information, 3 = Print detected words

# Mute OpenAI. 1 = Do not listen for OpenAI queries
SOFT_MUTE = 0


# ############################################################################
#  Write logMessage messages
# ############################################################################

def logMessage(level, message):
    if level <= LOG_LEVEL:
        print(message)


# ############################################################################
#  Error output via audio
# ############################################################################

def errorOut(errorCode):
    if errorCode is None:
        return False
    if errorCode in CONFIG['messages']:
        logMessage(1, CONFIG['messages'][errorCode])
        playAudioMessage(errorCode)
        return True
    else:
        return False


# ############################################################################
#  Play preconfigured audio message
# ############################################################################

def playAudioMessage(messageKey):
    if messageKey in CONFIG['messages']:
        textToSpeech(CONFIG['messages'][messageKey], messageKey)
    else:
        logMessage(1, f"Message key {messageKey} not found in configuration")


# ############################################################################
#  Read configuration from file
# ############################################################################

def readConfig(configFile):
    global CONFIG

    try:
        if not os.path.isfile(configFile):
            raise FileNotFoundError(f"Config file {configFile} not found.")

        logMessage(1, f"Reading config file {configFile} ...")
        CONFIG.read(configFile)

        # HomeAI won't work without API credentials
        if CONFIG['OpenAI']['openAIKey'] == 'none':
            raise ValueError("Open AI API key not configured")
        if CONFIG['AWS']['awsKeyId'] == 'none':
            raise ValueError("AWS key id not configured")
        if CONFIG['AWS']['awsKeySecret'] == 'none':
            raise ValueError("AWS key not configured")

        openai.api_key = CONFIG['OpenAI']['openAIKey']      
        CONFIG['messages']['welcome'] = CONFIG['messages']['welcome'].format(activationWord=CONFIG['common']['activationWord'])

        logMessage(2, "Control commands:")
        for controlWord,commandList in CONFIG.items('commands'):
            logMessage(2, f"{controlWord}: {commandList}")

        return True
    
    except ValueError as err:
        logMessage(0, err)
    except FileNotFoundError as err:
        logMessage(0, err)

    return False    


# ############################################################################
#  Save audio to file
#
#    audio: audio data
#    name: absolute name of audio file
# ############################################################################

def saveRecordedAudio(audio, name):

    audioData = audio.get_raw_data()
    with wave.open(name, "wb") as wavFile:
        wavFile.setnchannels(CHANNELS)  # Mono
        wavFile.setsampwidth(BYTES_PER_SAMPLE)  # 2 bytes per sample
        wavFile.setframerate(audio.sample_rate)  # Use original sample rate
        wavFile.writeframes(audioData)
        wavFile.close()


# ############################################################################
#  Listen for activation word
#  Will return activation word or detected command.
# ############################################################################

def listenForActivationWord(recognizer, microphone):

    activationWord = CONFIG['common']['activationWord'].lower()
    listenTime = CONFIG['common']['duration']
    recFile = CONFIG['common']['audiofiles'] + "/commandrec.wav"

    # Listen
    try:
        with microphone as source:
            logMessage(2, f"Listening for {listenTime} seconds for activation word {activationWord} ...")
            audio = recognizer.listen(source, timeout=float(listenTime))
            #audio = recognizer.record(source, duration=float(listenTime))

        if LOG_LEVEL == 3:
            saveRecordedAudio(audio, recFile)

        # Speech recognition
        result = recognizer.recognize_google(audio, language=CONFIG['Google']['language'])
        logMessage(3, "Understood: " + result)
        words = result.lower().split()
        logMessage(3, words)

        # Search for activation word. Will raise a ValueError exception if activation word is not found
        idxActivationWord = words.index(activationWord)
        logMessage(3, "Understood activation word " + activationWord)

        # Check for control commands
        if len(words) > idxActivationWord+1:
            for controlWord,commandList in CONFIG.items('commands'):
                commandWords = commandList.split(',')
                if words[idxActivationWord+1] in commandWords:
                    logMessage(3, "Understood control command " + commandList + " [" + controlWord + "]")
                    return controlWord

        return activationWord

    except ValueError:   # Raised by index()
        logMessage(2, "Value Error: List of words does not contain activation word " + activationWord)
    except LookupError:
        logMessage(1, "Lookup Error: Could not understand audio")
        return "didNotUnderstand"
    except sr.UnknownValueError:
        logMessage(2, "Unknown Value Error: No input or unknown value")
    except sr.WaitTimeoutError:
        logMessage(2, "Listening timed out")

    return None


# ############################################################################
#  Listen for OpenAI command
# ############################################################################

def listenForOpenAICommand(recognizer, microphone):

    listenTime = CONFIG['common']['duration']
    recFile = CONFIG['common']['audiofiles'] + "/openairec.wav"

    try:
        # Listen
        with microphone as source:
            logMessage(2, f"Listening for query for {listenTime} seconds ...")
            audio = recognizer.listen(source, timeout=float(listenTime))

        saveRecordedAudio(audio, recFile)

        # Convert speech to text
        prompt = speechToText(recFile)
        if prompt == "":
            logMessage(1, "Couldn't understand the command")
            return 'didNotUnderstand'
        else:
            return prompt
    
    except sr.UnknownValueError:
        logMessage(1, "Couldn't understand the command")
        return 'didNotUnderstand'
    except sr.WaitTimeoutError:
        logMessage(2, "No input")

    return None


# ############################################################################
#  Convert speech to text with OpenAI Whisper
# ############################################################################

def speechToText(recFile):

    audioFile = open(recFile, "rb")
    text = openai.Audio.transcribe("whisper-1", audioFile, language=CONFIG['OpenAI']['openAILanguage'])
    audioFile.close()

    logMessage(3, text)
    prompt = text['text']
    logMessage(3, prompt)

    return prompt


# ############################################################################
#  Ask Chat GPT
# ############################################################################

def askChatGPT(prompt):

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=CONFIG['OpenAI']['openAIModel'],
        messages=messages,
        temperature=0
    )
    return response.choices[0].message["content"]


# ############################################################################
#  Play an audio file
#
#    loops = -1: play endlessly
#    loops = 0: play once
# ############################################################################

def playAudioFile(fileName, background=False, loops=0):

    if not os.path.isfile(fileName):
        found = False
        if not fileName.startswith(CONFIG['common']['audiofiles']):
            fileName = CONFIG['common']['audiofiles'] + "/" + fileName
            if os.path.isfile(fileName):
                found = True
        if not found:
            logMessage(2, f"Can't play audio file {fileName}. File not found.")
            return
    
    pygame.mixer.init()
    pygame.mixer.music.load(fileName)
    pygame.mixer.music.play(loops)

    if not background:
        # Wait until the audio playback is completed
        while pygame.mixer.music.get_busy():
            pass


# ############################################################################
#  Play an audio PCM stream
# ############################################################################

def playAudioStream(stream):

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(BYTES_PER_SAMPLE),
        channels=CHANNELS,
        rate=int(CONFIG['common']['sampleRate']),
        output=True)

    with closing(stream) as pollyStream:
        while True:
            data = pollyStream.read(READ_CHUNK)
            if data is None or len(data) == 0:
                break
            stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


# ############################################################################
#  Fade out audio
#
#    duration: fade out duration in seconds
# ############################################################################

def fadeOutAudio(duration):
    pygame.mixer.music.fadeout(duration * 1000)


# ############################################################################
#  Convert text to speech with AWS Polly and play result
#
#    outputFile: Name of temporary audio file. File will be created or is
#       expected to be found in "audiofiles" directory. Name must be specified
#       without file extension.
#    useCache: Flag for using cached/existing file. Set it to False to force
#       creation of a new audio file
#    background: Flag for playing audio in background. Is ignored if no
#       outputFile is specified
# ############################################################################

def textToSpeech(text, outputFile=None, useCache=True, background=False):

    session = boto3.Session(
        aws_access_key_id=CONFIG['AWS']['awsKeyId'],
        aws_secret_access_key=CONFIG['AWS']['awsKeySecret'],
        region_name=CONFIG['AWS']['region']
    )
    polly = session.client('polly')

    # Determine audio output format
    if outputFile is None:
        format = "pcm"
        sampleRate="16000"
    else:
        format = "mp3"
        sampleRate="22050"
        fileName = CONFIG['common']['audioFiles'] + "/" + outputFile + "." + format

    try:
        # Convert text to audio stream
        response = polly.synthesize_speech(
            Engine='standard',
            Text=text,
            OutputFormat=format,
            VoiceId=CONFIG['AWS']['pollyVoiceId'],
            LanguageCode=CONFIG['AWS']['language'],
            SampleRate=sampleRate
        )

    except (BotoCoreError, ClientError) as error:
        logMessage(0, "BotoCoreError" + error)
        return

    # Output stream
    if outputFile is None:
        playAudioStream(response['AudioStream'])
    else:
        if not os.path.isfile(fileName) or not useCache:
            # Write stream to file
            logMessage(2, "Writing speech audio to file " + fileName)
            with open(fileName, 'wb') as f:
                f.write(response['AudioStream'].read())
        playAudioFile(fileName, background=background)


# ############################################################################
#  List configured microphones
# ############################################################################

def listMicrophones():

    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    print("Available microphone devices:")
    for i in range(0, numdevices):
        dev = p.get_device_info_by_host_api_device_index(0, i)
        if (dev.get('maxInputChannels')) > 0:
            print("Input Device id ", dev.get('index'), " - ", dev.get('name'))
    p.terminate()


# ############################################################################
#  Select microphone
# ############################################################################

def selectMicrophone(micName):

    deviceIndex = None

    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        dev = p.get_device_info_by_host_api_device_index(0, i)
        if (dev.get('maxInputChannels')) > 0 and micName in dev.get('name'):
            # Found microphone
            deviceIndex = dev.get('index')
            print("Selected microphone ", dev.get('name'))
            break
    p.terminate()

    return deviceIndex


# ############################################################################
#  Main function
# ############################################################################

def main():
    global LOG_LEVEL, SOFT_MUTE

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog="HomeAI", description="Home AI Assistant")
    parser.add_argument("--config", default="homeai.conf", help="Name of configuration file")
    parser.add_argument("--list_microphones", action="store_true", help="List available microphones")
    parser.add_argument("--microphone", help="Set name of microphone")
    parser.add_argument("--log_level", default=0, type=int, choices=range(0, 4), help="Set level of log messages")
    parser.add_argument("--no_welcome", action="store_true", help="Do not play welcome message")
    parser.add_argument("--version", action="version", version='%(prog)s ' + VERSION)
    args = parser.parse_args()

    # List available microphones
    if args.list_microphones:
        listMicrophones()
        return

    LOG_LEVEL = int(args.log_level)
    print("Set log level to " + str(LOG_LEVEL))

    # Read configuration
    if not readConfig(args.config):
        return

    # Setup microphone
    deviceIndex = None
    if args.microphone:
        deviceIndex = selectMicrophone(args.microphone)
    else:
        print("Using system default microphone")
    microphone = sr.Microphone(sample_rate=int(CONFIG['common']['sampleRate']), device_index=deviceIndex)

    # Setup recognizer
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    if int(CONFIG['common']['energyThreshold']) == -1:
        logMessage(2, "Calibrating energy threshold ...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        logMessage(2, "Speech recognition energy threshold = " + str(recognizer.energy_threshold))
    else:
        recognizer.energy_threshold = CONFIG['common']['energyThreshold']

    # Output welcome message. Will be cached in welcome.mp3
    if not args.no_welcome:
        playAudioMessage('welcome')
        playAudioFile("listening.wav")

    while True:
        # Listen for activation word
        command = listenForActivationWord(recognizer, microphone)

        if command == 'stop':
            logMessage(2, "Stopping audio playback")
            fadeOutAudio(1)
        elif command == 'mute':
            logMessage(2, "Muted")
            SOFT_MUTE = 1
        elif command == 'unmute':
            logMessage(2, "Unmuted")
            SOFT_MUTE = 0
        elif command == 'terminate':
            logMessage(0, "Shutting down home-ai")
            playAudioMessage('shutdown')
            break
        elif command == CONFIG['common']['activationWord'].lower():
            if SOFT_MUTE:
                errorOut("muted")
            else:
                playAudioFile("listening.wav", background=True)
                logMessage(2, ">>> Ask Open AI")

                # Listen for ChatGPT query
                prompt = listenForOpenAICommand(recognizer, microphone)

                if not errorOut(prompt):
                    try:
                        # Play sound until response from ChatGPT arrived and is converted to audio
                        playAudioFile("processing.wav", loops=-1, background=True)
                        
                        # Send query to Chat GPT and output response
                        response = askChatGPT(prompt)
                        logMessage(3, response)
                        fadeOutAudio(1)
                        textToSpeech(response, "response", useCache=False, background=True)

                    except Exception:
                        fadeOutAudio(1)
                        errorOut("genericError")

        elif not command is None:
            if not errorOut(command):
                logMessage(1, "Unknown command " + command)


if __name__ == "__main__":
    main()
