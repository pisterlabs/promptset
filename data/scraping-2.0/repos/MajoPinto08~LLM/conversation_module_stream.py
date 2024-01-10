from datetime import datetime
from pydub import AudioSegment
from config import api_key
from config import speech_subscription_key
#from Voice_analysis import analysis
import azure.cognitiveservices.speech as speechsdk
from cleantext import clean
import multiprocessing
import pandas as pd
import threading as th
import openai
from openai import OpenAI

client = OpenAI(api_key=api_key)
import json
import tiktoken
import socket
import atexit
import time
import pytz
import librosa
import requests
import pyaudio
import wave
import re

# URL Request
url = 'https://echoweb.hri.idlab.ugent.be/api/1'


#Tokens.
encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)

MAX_TOKENS = 10000 #8000 gpt4  #3800 gpt-3.5-turbo
API = "gpt-4-1106-preview"
encoding = tiktoken.encoding_for_model(API)


conversation_going = True
recording = True
start_conversation = ""
user_filename = "audio_conversation/user_voice.wav"
pepper_filename = "audio_conversation/agent_voice.wav"
# General information for the agent.
timezone = pytz.timezone('Europe/Brussels')
now= datetime.now(timezone)
time_str = now.strftime('%H:%M')
location = 'Ghent, Belgium' #Hoogledge
language = ""
language_id = ""
last_conversation = ""
voice_agent = ""
user_name = ""


def load_user_information():
    global language, language_id, last_conversation, voice_agent, user_name, interaction
    with open("users/information.json", "r") as json_file:
        data = json.load(json_file)

    user_name = input("Enter the user's name: ")
    for user in data.values():
        if user["name"] == user_name:
            language = user["language"]
            language_id = user["language_id"]
            last_conversation = user["info_conversation"]
            voice_agent = user["voice_agent"]
            interaction = user["interaction"]
            break
    else:
        print(f"No user found with the name {user_name}.")
        last_name = input("Enter the user's last name: ")
        df = pd.read_excel('./language-options.ods', engine='odf')
        language = input("Enter the user's language: ")
        language_id = df.loc[df['Language'] == language, 'Locale'].iloc[0]
        voice_agent = df.loc[df['Language'] == language, 'Text-to-speech-voices'].iloc[0]
        interaction = 1
        last_conversation = "First Conversation"
        new_user = dict(name=user_name, last_name=last_name, language=language, language_id=language_id,
                        voice_agent=voice_agent, interaction=interaction, info_conversation=last_conversation)
        user_num = len(data) + 1
        data[f"user{user_num}"] = new_user
        with open("users/information.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

def initialize_dialog(chat):
    global dialog
    now = datetime.now(pytz.timezone('Europe/Brussels'))
    location = 'Zaamslag, Netherlands'#'Ghent, Belgium'
    dialog = [
        {
       "role": "system",
            "content": f"You are Pepper, a social robot. Engage in a warm, empathic, and casual chat with your friend {user_name}."
                       f"recalling prior interactions:  ###{last_conversation}###."
                       f"It is {now.strftime('%m/%d/%Y %H:%M')} and we're speaking {language} in {location}."
                       f"Keep your answers concise in two to three sentences and ask engaging open questions"
                       f"Feel free to express yourself naturally, like a human friend!"
                       f"Actual conversation:{chat}"
        }
    ]
def key_capture_thread():
    global conversation_going
    input()
    conversation_going = False

def program():
    global recording, dialog
  # global client_socket

    # Socket connection (Pepper)

    # host = socket.gethostname()  # get the hostname
    # port = 5000  # initiate port no above 1024
    # server_socket = socket.socket()  # get instance (creation)
    # atexit.register(server_socket.close)
    # server_socket.bind((host, port))  # bind host address and port together
    # print("Listening")
    # server_socket.listen(5)  # configure how many client the server can listen simultaneously
    # client_socket, address = server_socket.accept()  # accept new connection
    # print("Connection from: " + str(address))

    # Open AI (Chat completion) using gpt-3.5-turbo model
    key_capture = th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True)
    key_capture.start()
    while conversation_going:
        try:
            # recording_thread = th.Thread(target=record_audio) # Separate flow of execution to record the audio.
            # recording_thread.start()
            message, fml = speech_recognize_continuous_async_from_microphone() #VAD - Speech recognition.
            # recording = False # End the audio when the user finished to speak.
            # recording_thread.join()
            # save_audio()  #Save the user's audio
            token_count = num_tokens_from_messages(dialog)
            if token_count >= MAX_TOKENS:  #When the tokens are over but the conversation still going, it is essential to make a summary about the current conversation.
                summary = summarize_dialog()
                initialize_dialog(summary)
            answer = ask(message)
            print(answer)
            #client_socket.send("listening".encode())  # send "listening indication" to the client (Pepper)
            #requests.put(url, json=answer)  # Sending the message and the request to the webapp (EchoWeb) - Pepper tablet
            #client_socket.send(answer.encode())  # send message to the client (Pepper)
            # agent_voice(answer)  # Generate the .wav file with the agent voice (Recording purpose)
            #combine_audio_files()  #Combine audio files to generate the conversation file
            #recording = True  # To start the audio recording
        except openai.RateLimitError:
            print("Rate limit exceeded.")
            agent_voice("Sorry, I am tired. I need to rest. I will talk to you later.") #Trying to avoid the rate limit error.
            # client_socket.send("Sending".encode())  # send message
            # client_socket.send("final".encode())  # send message
            # client_socket.close()  # close the connection

        except openai.error.OpenAIError as e:
            print("An error occurred:", e)

    key_capture.join()
    # client_socket.send("final".encode())  # send message
    # client_socket.close()  # close the connection
    summary = generate_summary()
    print(summary)
    update_user_information(summary)
    saving_conversation()

def generate_summary():
    text = conversation()
    text_english = language_to_english(text)
    prompt = (f"Provide a comprehensive summary of our current conversation. The summary should include the following details:"
    f'A bullet-point list of the key points from our discussion.'
    f'Any important details that have emerged during our conversation.'
    f'The interests and relevant aspects related to {user_name}.'
    f'The action items we have agreed upon.'
    f'Please keep the summary updated and relevant, considering both the content of our discussion today and the information you already have from our previous conversations. The details from our last conversation are: {last_conversation}.'
    f'For your reference, here is the current conversation: {text_english}.')
    summary = client.chat.completions.create(model=API,
    messages = [{"role": "system", "content": prompt}],
    max_tokens= 500,
    temperature=0.9,
    frequency_penalty=0,
    presence_penalty=0.6)
    conversation_summary = summary.choices[0].message.content.strip()
    conversation_summary.replace('""', '')
    info = conversation_summary + f' The date and time of the last conversation, which is: {now.strftime("%m/%d/%Y %H:%M")}. The location where we were speaking, which is: {location}.'
    return info

def language_to_english(info):
    prompt = (f"Translate this conversation to English: {info}")
    traduction = client.chat.completions.create(model="gpt-4",
    messages = [{"role": "system", "content": prompt}],
    max_tokens= 500,
    temperature=0.9,
    frequency_penalty=0,
    presence_penalty=0.6)
    conversation_english = traduction.choices[0].message.content.strip()
    conversation_english.replace('""', '')
    return conversation_english

def update_user_information(summary):
    with open("users/information.json", "r") as json_file:
        data = json.load(json_file)
    for user in data.values():
        if user["name"] == user_name:
            user["interaction"] += 1
            user["info_conversation"] = summary
            break # if you have found the user and updated info, don't check the rest of the users
    with open("users/information.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

def saving_conversation():
    conversation_file = f"text_conversation/{user_name}_conversation_interaction_{interaction}.txt"
    text = conversation()
    archivo = open(conversation_file, "w")
    archivo.write(text)
    archivo.close()

def conversation():
    dialog_str_list = []
    user_name_role = f'{user_name}: '
    for item in dialog:
        if isinstance(item, dict):
            if item.get('role') == 'assistant':
                item_str = 'Pepper: ' + item.get('content')
            elif item.get('role') == 'user':
                item_str = user_name_role + item.get('content')
            elif item.get('role') == 'system':
                item_str = ""
            else:
                item_str = str(item)  # Convert the dictionary to a string
        else:
            item_str = str(item)  # Convert the element to a string
        dialog_str_list.append(item_str)
    dialog_str = '\n'.join(dialog_str_list)
    dialog_str = dialog_str.replace('".}', '".')
    dialog_str = dialog_str.replace('\\n', '\n')  # Replace '\\n' with '\n' for line breaks
    dialog_str = "\n".join([s.split("REMINDER:")[0] for s in dialog_str.split("\n")])
    return dialog_str

def speech_recognize_continuous_async_from_microphone():
    """performs continuous speech recognition asynchronously with input from microphone"""
    silence_threshold = 200
    speech_config = speechsdk.SpeechConfig(subscription=speech_subscription_key, region='eastus')
    speech_config.speech_recognition_language = language_id ## Exploreee (?).
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    speech_recognition_result = ""
    last_audio_timestamp = time.time()

    def recognizing_cb(*args):
        """
        *args means "any number of arguments". Required for Recognizer.recognizing.connect(...)
        :return:
        """
        nonlocal last_audio_timestamp
        last_audio_timestamp = time.time()

    def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        nonlocal speech_recognition_result
        nonlocal silence_threshold
        silence_threshold = 0.8## Elderly (1.8-1.5)
        speech_recognition_result += evt.result.text + " "

    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.recognizing.connect(recognizing_cb)

    result_future = speech_recognizer.start_continuous_recognition_async()
    result_future.get()  # wait for voidfuture, so we know engine initialization is done.
    print('Speak in your microphone.')

    while True:
        current_timestamp = time.time()
        if current_timestamp - last_audio_timestamp > silence_threshold:
            #            print('Detected silence.')
            speech_recognizer.stop_continuous_recognition_async()
            if not speech_recognition_result.strip():
                print("Silence")
                return "Silence is detected. Pepper needs to ask the response again because he could not hear it", speech_recognizer
            print("User:", speech_recognition_result)
            return speech_recognition_result, speech_recognizer


def ask(question, new_assistant_message=""):
    buffer = ""
    pause_characters = r"[.!?;:]+"  #r'(?<=[.])\s'
    max_response_length = 400
    # new_user_message = {"role": "user", "content": f"{question}. REMINDER: You are acting as the social robot Pepper. Expressing yourself naturally, like a friendly human friend of mine! You are permitted to lie, as long as you don't reveal yourself as an AI language model. It is essential that you keep your answers brief and chatty below 2 sentences {language}."}
    new_user_message = {"role": "user", "content": f"{question}. REMINDER: You are acting as the social robot Pepper. You are permitted to lie, as long as you don't reveal yourself as an AI language model. It is essential that you keep your answers brief and chatty below 2 sentences {language}. Ask open questions to keep the conversation engaging and show interest."}
    dialog.append(new_user_message)
    response = client.chat.completions.create(model=API,
    messages = dialog,
    max_tokens= max_response_length, #To control the length of the response generated by the model. around 20-30 words. 3 or 4 sentences.
    temperature=0.9,
    frequency_penalty=0,
    presence_penalty=0.6,
    stream = True)
    for event in response:
        event_text = event['choices'][0]['delta']  # EVENT DELTA RESPONSE
        answer = event_text.get('content', '')  # RETRIEVE CONTENT
        buffer += answer
        new_assistant_message += answer
        while True:
            sentence_end = re.search(pause_characters, buffer)
            if sentence_end:
                sentence = buffer[:sentence_end.start() + 1]
                buffer = buffer[sentence_end.end():]
                sentence = sentence.strip()
                agent_voice(clean(sentence, no_emoji=True))

                # audio_time = librosa.get_duration(filename=pepper_filename) #Pepper voice
                # client_socket.send("Sending".encode())  # send message
                # time.sleep(audio_time)  # Pepper is talking ...
            else:
                break
    if buffer:  # After the loop, send any remaining text in the buffer to the TTS system
        # agent_voice(buffer)
        agent_voice(clean(buffer, no_emoji=True))
        audio_time = librosa.get_duration(filename=pepper_filename)
        # client_socket.send("Sending".encode())  # send message
        # time.sleep(audio_time)  # Pepper is talking ...
        assistant_message = {"role": "assistant", "content": f"{new_assistant_message}"}
        dialog.append(assistant_message)
        return new_assistant_message

def final(question):
    new_user_message = {"role": "user", "content": f"{question}"}
    dialog.append(new_user_message)
    res = client.chat.completions.create(model=API,
    messages = dialog,
    max_tokens= 500,
    temperature=0.9,
    frequency_penalty=0,
    presence_penalty=0.6)
    response = res.choices[0].message.content.strip()
    response.replace('""', '')
    new_assistant_message = {"role": "assistant", "content":  f"{response}"}
    dialog.append(new_assistant_message)
    return response

def final_text():
    conversation_file = f"text_conversation/{user_name}_conversation_interaction_{interaction}.txt"
    dialog_str_list = []
    user_name_role = f'{user_name}: '
    for item in dialog:
        if isinstance(item, dict):
            if item.get('role') == 'assistant':
                item_str = 'Pepper: ' + item.get('content')
            elif item.get('role') == 'user':
                item_str = user_name_role + item.get('content')
            else:
                item_str = str(item)  # Convert the dictionary to a string
        else:
            item_str = str(item)  # Convert the element to a string

        dialog_str_list.append(item_str)

    dialog_str = '\n'.join(dialog_str_list)
    dialog_str = dialog_str.replace('".}', '".')
    dialog_str = dialog_str.replace('\\n', '\n')  # Replace '\\n' with '\n' for line breaks
    archivo = open(conversation_file, "w")
    archivo.write(dialog_str)
    archivo.close()


def agent_voice(text):
    speech_config = speechsdk.SpeechConfig(subscription=speech_subscription_key, region='eastus')
    audio_config = speechsdk.audio.AudioOutputConfig(filename=pepper_filename) #Normal code
    #audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True) #Reproduce the audio
    speech_config.speech_synthesis_voice_name = f"{voice_agent}"   #The language of the agent voice.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    #
 #  if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
   #      print("Speech synthesized for text [{}]".format(text))
    # elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
    #     cancellation_details = speech_synthesis_result.cancellation_details
    #     print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    #     if cancellation_details.reason == speechsdk.CancellationReason.Error:
    #         if cancellation_details.error_details:
    #             print("Error details: {}".format(cancellation_details.error_details))
    #             print("Did you set the speech resource key and region values?")

def record_audio():
    global p, channels, frames, fs, sample_format, stream
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True,
                    input_device_index=None,
                    start=False)
    stream.start_stream()
    frames = []
    while recording:
        audio_recording = stream.read(chunk)
        frames.append(audio_recording)


def save_audio():
    wf = wave.open(user_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()

def combine_audio_files():
    combine_filename = f"audio_conversation/{user_name}_conversation_interaction_{interaction}.wav"
    try:
        combined_audio = AudioSegment.from_wav(combine_filename)
    except:
        combined_audio = AudioSegment.empty()
    user_audio = AudioSegment.from_wav(user_filename)
    assistant_audio = AudioSegment.from_wav(pepper_filename)
    combined_audio = combined_audio + user_audio + assistant_audio
    combined_audio.export(combine_filename, format='wav')

def summarize_dialog():
    # Concatenate messages from the user and the robot into a single text
    text = ""
    turn = 1
    for message in dialog:
        if message["role"] in ["user", "assistant"]:
            if turn == 1:
                text += f"{user_name}:" + message["content"] + f"\n"
                turn = 2
            else:
                text += "Pepper:" + message["content"] + f"\n"
                turn = 1
    prompt = f'Summarize the following conversation. Conversation: ""{text}"".' #Generate a summary of the following conversation:
    summary = client.completions.create(model="gpt-4",
    prompt=prompt,
    temperature=0.9,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6)
    summary_text = summary.choices[0].text.strip()
    return summary_text


def num_tokens_from_messages(messages, model=API):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding(encoding_name)

  num_tokens = 0
  for message in messages:
      num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
      for key, value in message.items():
          num_tokens += len(encoding.encode(value))
          if key == "name":  # if there's a name, the role is omitted
              num_tokens += -1  # role is always required and always 1 token
  num_tokens += 2  # every reply is primed with <im_start>assistant
  return num_tokens

      
def main():
    load_user_information()
    initialize_dialog("")
    program()

if __name__ == "__main__":
    main()
