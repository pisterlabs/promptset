import requests
import numpy as np
import os
import openai
import speech_recognition as sr
from urllib.error import URLError
from dotenv import load_dotenv
import time
import tiktoken


# setup
load_dotenv() # load environment variables from .env file
sleep_time = 0.1 # in seconds
sampling_frequency = 16000 # 16 kHz
number_of_samples_per_chunk = 1365 
time_between_audio_chunks = number_of_samples_per_chunk / sampling_frequency # in seconds
corrected_time_between_audio_chunks = time_between_audio_chunks*0.8 # considering other delays
max_response_tokens = 250
token_limit = 4096
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
BODY_URL = "http://localhost:5004"  # assuming it runs locally


class NaoStream:
    
    def __init__(self, audio_generator):
        self.audio_generator = audio_generator

    def read(self, size=-1):  # added size parameter, default -1
        try:
            return next(self.audio_generator)
        except StopIteration:
            return b''
            

class NaoAudioSource(sr.AudioSource):

    def __init__(self, server_url=BODY_URL):
        self.server_url = server_url
        self.stream = None
        self.is_listening = False
        self.CHUNK = 1365 # number of samples per audio chunk
        self.SAMPLE_RATE = 16000 # 16 kHz
        self.SAMPLE_WIDTH = 2 # each audio sample is 2 bytes

    def __enter__(self): # this is called when using the "with" statement
        requests.post(f"{self.server_url}/start_listening")
        self.is_listening = True
        self.stream = NaoStream(self.audio_generator())  # wrap the generator
        return self # return object (self) to be used in the with statement

    def audio_generator(self): # generator function that continuously fetches audio chunks from the server as long as 'self.is_listening' is True
   
        while self.is_listening:
            response = requests.get(f"{self.server_url}/get_audio_chunk")
            yield response.content # yield is used to return a value from a generator function, but unlike return, it doesn't terminate the function -> instead, it suspends the function and saves its state for later resumption
            current_buffer_length = requests.get(f"{self.server_url}/get_server_buffer_length").json()["length"]
            correcting_factor = 1.0 / (1.0 + np.exp(current_buffer_length - np.pi)) # if buffer becomes long, the time between audio chunks is decreased
            corrected_time_between_audio_chunks = time_between_audio_chunks * correcting_factor
            time.sleep(corrected_time_between_audio_chunks) # wait for the next audio chunk to be available

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_listening = False
        requests.post(f"{self.server_url}/stop_listening")


def get_user_text():

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1  # seconds of non-speaking audio before a phrase is considered complete
    recognizer.operation_timeout = 4  # increasing the timeout duration
    audio_data = None
    filename = "input.wav"

    while True:
        # record audio only if it hasn't been recorded yet
        if audio_data is None:
            with NaoAudioSource() as source:
                print("Recording...")
                start_time = time.time()
                audio_data = recognizer.listen(source, phrase_time_limit=10, timeout=None)
                with open(filename, "wb") as f:
                    f.write(audio_data.get_wav_data())
                print(f"Recording took {time.time() - start_time} seconds")

         # transcribe audio to text
        try:
            print("Transcribing...")
            start_time = time.time()
            text = recognizer.recognize_google(audio_data)
            print(f"Transcribing took {time.time() - start_time} seconds")
            print("You said: " + text)
            return text
        except (sr.RequestError, URLError, ConnectionResetError) as e:
            print(f"Network error: {e}, retrying after a short delay...")
            time.sleep(sleep_time)  # adding a delay before retrying
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio, retrying...")
            audio_data = None  # reset audio_data to record again
        except TimeoutError as e:
            print(f"Operation timed out: {e}, retrying after a short delay...")
            audio_data = None  # reset audio_data to record again



def get_gpt_text(conversation_context):

    # trim the conversation context to fit the token limit
    conversation_context = trim_context(conversation_context)

    # process the received input with GPT
    start = time.time()
    response = openai.ChatCompletion.create(
        engine="NAO35",
        messages=conversation_context
    )
    end = time.time()
    print(f"{response.engine} took {end - start} seconds to respond")

    # xtract the GPT response
    gpt_message = response['choices'][0]['message']['content']

    print(f"Nao: {gpt_message}")

    return gpt_message


def send_gpt_text_to_body(gpt_message):

    requests.post(f"{BODY_URL}/talk", json={"message": gpt_message}) # send the GPT response to the body


def save_conversation(context, filename):

    with open("conversation_context.txt", "w") as f:
        for entry in conversation_context:
            role = entry['role'].capitalize()  # capitalize the role for formatting
            content = entry['content']
            f.write(f"{role}:\n{content}\n\n")


def trim_context(context):
    """see https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python-new&pivots=programming-language-chat-completions for more details."""

    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    conv_history_tokens = num_tokens_from_messages(context)
    while conv_history_tokens + max_response_tokens >= token_limit:
        del context[1] 
        conv_history_tokens = num_tokens_from_messages(context)

    return context


# conversation loop  ====================================================================================================

with open("system_prompt.txt", "r") as f:
    system_prompt = f.read() # read system prompt from file

conversation_context = [{"role": "system", "content": system_prompt}] # initialize conversation context with system prompt

running = True
while running:
    user_message = get_user_text() # get the user's message
    conversation_context.append({"role": "user", "content": user_message}) # add the user's message to the conversation context
    gpt_message = get_gpt_text(conversation_context)
    send_gpt_text_to_body(gpt_message)
    conversation_context.append({"role": "assistant", "content": gpt_message}) # add the GPT-4 response to the conversation context
    save_conversation(context=conversation_context, filename="conversation_context.txt") # write conversation context to file for easier debugging etc.