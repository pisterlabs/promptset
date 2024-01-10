import os
import openai
import pickle
from pvrecorder import PvRecorder
from pvcheetah import CheetahActivationLimitError, create
from elevenlabs import generate, play, set_api_key, Voices
#from elevenlabs.api import Voices # enable to change voice
from dotenv import load_dotenv  

## LOAD ENVIRONMENT VARIABLES
load_dotenv()
# get API/access keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
cheetah_access_key = os.getenv("PV_ACCESS_KEY")

## OPENAI CONFIG
openai_model = 'gpt-3.5-turbo' #gpt-3.5-turbo, gpt-4
openai_temperature = 0.7
openai_max_tokens = 150

## CHEETAH CONFIG
library_path = None
model_path = None
endpoint_duration_sec = int(1.5)  # seconds to wait for speech to end before processing
disable_automatic_punctuation = False
show_audio_devices = False  # set True to show list of audio devices
if show_audio_devices:
    for index, name in enumerate(PvRecorder.get_available_devices()):
        print('Device #%d: %s' % (index, name))
audio_device_index = 0

## ELEVENLABS VOICE CONFIG
# Set API key
set_api_key(elevenlabs_api_key) # type: ignore
## get voice index from API if you want to change voice and save to pickle file
#voices = Voices.from_api()
#my_voice = voices[29]
file_path = 'saved_voice.pkl'
## Serialize and save the object to the file
#with open(file_path, 'wb') as file:
#    pickle.dump(my_voice, file)
# Load voice object from pickle file
with open(file_path, 'rb') as file:
    my_voice_loaded = pickle.load(file)
# configure elevenlabs voice
#my_voice_loaded.settings.stability = 0.5
#my_voice_loaded.settings.similarity_boost = 0.75

def ask_chatgpt(question, chat_log):
    if chat_log is None:
        # Set API key            
        openai.api_key = openai_api_key
        # Initialize chat log
        chat_log = [{
            'role': 'system',
            'content': 'You are helpful assistant that can always assist with an answer. You answer in less than 30 words when possible.'
        }]
    # Add question to chat log
    chat_log.append({'role': 'user', 'content': question})
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=chat_log,
        temperature=openai_temperature,
        max_tokens=openai_max_tokens,
    )
    # Add answer to chat log
    answer = response['choices'][0]['message']['content'] # type: ignore
    chat_log.append({'role': 'assistant', 'content': answer})

    return answer, chat_log

def text_to_speech(text):
    # Generate audio from API
    audio = generate( # type: ignore
        text=text,
        voice=my_voice_loaded,
        model="eleven_monolingual_v1"
    )
    # play audio
    audio: bytes = audio  # audio is Union so turn into bytes (to surpress annoying errors)
    play(audio)
    
    return

def main():

    # Create an instance of cheetah
    cheetah = create(
        access_key=cheetah_access_key, # type: ignore
        library_path=library_path,
        model_path=model_path,
        endpoint_duration_sec=endpoint_duration_sec,
        enable_automatic_punctuation=disable_automatic_punctuation
        )

    # chatgpt chatlog starts as none
    chat_log = None

    try:
        # Start recording
        recorder = PvRecorder(frame_length=cheetah.frame_length, device_index=audio_device_index)
        input("Press Enter to record another question...")
        print('Listening... (press Ctrl+C to stop)')
        

        try:
            while True:
                recorder.start()
                partial_transcript, is_endpoint = cheetah.process(recorder.read())
                print(partial_transcript, end='', flush=True)

                if is_endpoint:
                    # send question to chatgpt
                    question = cheetah.flush()
                    print(question)
                    
                    answer, chat_log = ask_chatgpt(question, chat_log)
                    print(answer)
                    # send answer to elevenlabs
                    text_to_speech(answer)
                    recorder.stop()
                    input("Press Enter to record another question...")
                    print('Listening... (press Ctrl+C to stop)')
        finally:
            print()
            recorder.stop()

    except KeyboardInterrupt:
        pass
    except CheetahActivationLimitError:
        print('AccessKey has reached its processing limit.')
    finally:
        cheetah.delete()

if __name__ == "__main__":
    main()
