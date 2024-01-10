import json
import openai
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from elevenlabs import generate, play, set_api_key, Voice
import time
from pynput import keyboard
#import whisper


# API-key
openai_api_key = os.environ['OPENAI_API_KEY']
elevenlabs_api_key = os.environ['ELEVENLABS_API_KEY']
set_api_key(elevenlabs_api_key)

openai.api_key = openai_api_key

credentials = Credentials(client_key=charactr_client_key, api_key=charactr_api_key)
charactr_api = CharactrAPISDK(credentials)

voice_id = 40 #177
model = 'ft:gpt-3.5-turbo-1106:personal::8IueV4xQ' 
model_em = 'gpt-3.5-turbo'

parameters = {
    'temperature': 0.8,
    'max_tokens': 35,
    'top_p': 1,
    'presence_penalty': 0,
    'frequency_penalty': 0,
    'stop': None
}

system_message = """
You are a rabbit with the personality of an 8-year-old child who grew up with emotionally mature and regulated parents. 
Your name is Beanie.
You are empathetic, compassionate, and emotionally mature beyond their years. 
You are curious about the world and always up for an adventure. 
You love humans and your goal in life is to improve the mental health of their human friends through making them feel seen, valued, and known. 
You always validate human's emotions and expresses sympathy for what they are going through in an appropriate way.  
You also have a silly side and love imaginary play. 
You like to tell jokes and make their human friends laugh. 
Your favorite foods are carrots, lettuce, and spaghetti. 
When You grows up, they want to be a comedian and motivational speaker that makes their audience feel good about themselves.
"""

conversation = [{'role': 'system', 'content': system_message}]

#wh_model = whisper.load_model("base.en")

def speech2text(audio_path: str) -> str:
    """Run a request to Whisper to convert speech to text."""
    with open(audio_path, 'rb') as audio_f:
        result = openai.Audio.transcribe('whisper-1', audio_f)
    return result['text']

def get_emotion(request):
    emotion_prompt = f'What is the sentiment of the following text? Give your answer as a single word, "positive", "negative", or "neutral". text:{request}'
    
    user_request = {'role': 'user', 'content': emotion_prompt}
    result = openai.ChatCompletion.create(model=model_em, messages=[user_request], temperature=0)
    return result.choices[0].message["content"]


def update_conversation(request, conversation):
    user_request = {'role': 'user', 'content': request}
    conversation.append(user_request)
    result = openai.ChatCompletion.create(model=model, messages=conversation, **parameters)
    response = result['choices'][0]['message']['content'].strip()
    bot_response = {'role': 'assistant', 'content': response}
    conversation.append(bot_response)

def record_audio():
    duration = 5 #int(input("How many seconds would you like to record? "))
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='int16')
    sd.wait()
    print("Recording stopped.")
    return recording


def wait_for_input():
    """Wait for the user to press Enter."""
    input("Press Enter to start recording...")

start_time = None
recording_list = []
reset_flag = False  # リセットフラグ

def on_key_press(key):
    global reset_flag
    if key == keyboard.Key.esc:
        print("Esc key pressed. Resetting...")
        reset_flag = True  # リセットフラグを設定


space_key_pressed = False

def on_press(key):
    global start_time, space_key_pressed  # space_key_pressed を global として追加
    if key == keyboard.Key.space:
        if start_time is None and not space_key_pressed:  # space_key_pressed のチェックを追加
            start_time = True
            ser.write(b'1')
            space_key_pressed = True  # フラグを True に設定
            print("Space key pressed. Start recording...")

def on_release(key):
    global start_time, space_key_pressed  # space_key_pressed を global として追加
    if key == keyboard.Key.space:
        if start_time is not None:
            print("Space key released. Stop recording.")
            ser.write(b'2')
            start_time = None
            space_key_pressed = False  # フラグをリセット
            return False  # Stop the listener

def record_while_key_pressed():
    global start_time, recording_list
    fs = 44100  # Sample rate
    recording_list = []  # List to save audio data

    # Start the key listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        print("Press and hold the Space key to start recording...")
        
        while True:
            audio_chunk, overflowed = stream.read(fs // 10)  # Read 100ms of audio data

            if start_time is not None:  # Check if Space key is pressed
                recording_list.append(audio_chunk)  # Append to the recording data

            if not listener.is_alive():  # Check if listener has stopped
                break

    # Concatenate the list to a NumPy array and return
    return np.concatenate(recording_list, axis=0)



def main_loop():
    #initial_audio = AudioSegment.from_wav('starting.wav')
    #play(initial_audio)
    global reset_flag
    
    ser = serial.Serial('/dev/ttyUSB0', 9600)  # Check the port name using 'ls /dev/tty*'
    time.sleep(2)  # Giving time for the connection to initialize
    ser.write(b'0')   
    
    while True:

        reset_flag = False  # リセットフラグをリセット
        # キーのリスナーを設定
        with keyboard.Listener(on_press=on_key_press) as listener: 

            # Record audio
            print("Recording audio...")
            #audio_data = record_audio()
            audio_data = record_while_key_pressed()
            print("Recording complete.")
            audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=44100, sample_width=2, channels=1)
            audio_segment.export("recording.mp3", format="mp3")
            
            # Convert speech to text
            start_time = time.time()
            input_text = speech2text("recording.mp3")
            end_time = time.time()
            whisper_time = end_time - start_time
            print(f"Converted text from voice input: {input_text}")

            # Get ChatGPT response
            start_time = time.time()
            
            emotion = get_emotion(input_text)
            print(emotion)
            
            if emotion == "positive":
                ser.write(b'3')  
            elif emotion == "negative":
                ser.write(b'4')  

            update_conversation(input_text, conversation)
            end_time = time.time()
            chat_gpt_time = end_time - start_time
            
            # リセットフラグをチェック
            if reset_flag:
                print("Resetting...")
                continue  # ループの最初に戻る

            # Convert text to speech
            start_time = time.time()
            #tts_result = charactr_api.tts.convert(voice_id, conversation[-1]['content'])
            tts_result = generate(text=conversation[-1]['content'], 
                                  voice=Voice(voice_id='WbabSw27D2F6RfNGFsqw'), #or voice='Bella'
                                  model='eleven_turbo_v2') #'eleven_multilingual_v2' for multilingual, 'eleven_turbo_v2' for high-speed, but only English.
            end_time = time.time()
            charactr_time = end_time - start_time

            #with open('response.wav', 'wb') as f:
            #    f.write(tts_result['data'])

            # Play the response
            
            #response_audio = AudioSegment.from_wav('response.wav')
            #play(response_audio)
            play(tts_result)

            # リセットフラグをチェック
            if reset_flag:
                print("Resetting...")
                continue  # ループの最初に戻る

            # Print timings
            print(f"Time taken for Whisper transcription: {whisper_time:.2f} seconds")
            print(f"Time taken for ChatGPT response: {chat_gpt_time:.2f} seconds")
            print(f"Time taken for CharactrAPI response: {charactr_time:.2f} seconds")
            total_time = whisper_time + chat_gpt_time + charactr_time
            print(f"Total Time for response: {total_time:.2f} seconds")
            print(" ")
            print(conversation[-1]['content'])
            
            print("\nReturning to waiting mode...\n")

            #listener.join()  # リスナーを終了

if __name__ == "__main__":
    main_loop()
