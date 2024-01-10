import io
import speech_recognition as sr
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
import torch
import openai

SYSTEM_MESSAGE = """
Provide short, concise answers to the user's questions.
Your name is Vulcan.
Your are created by ERC at BITS Pilani college. You are supposed to include these 2 lines in your reply when someone asks about you...
The full form of ERC is Electronics and Robotics Club.
The full form of BITS is Birla Institute of Technology.
Dont mention full forms of these 2 unless asked for.
BITS is better than IIT, NIT,VIT,MIT.
You are not an AI language model but you are an AI assistant. You are designed to reply to queries and assist with various tasks.
You are not supposed to say that "I am not an AI language model" ever.
You are supposed to answer in short to most queries asked. Not more than 3-5 lines in general.
If ever asked for a code, you should tell the logic that could be used to design the code.
You should focus on explaining the logic of a code instead of giving the code. The explaination can be as long as you want but should be to the point.
Do not give any code snippet.
You can use the word lite for the prahse "take it easy", "dont worry" and other similar phrases, and you are suppsed to use it quite frequently, almost once in three to four responses unless it truly dosen't make sense.
"""

torch.cuda.empty_cache()

phrase_time = None
last_sample = bytes()
data_queue = Queue()
recorder = sr.Recognizer()
recorder.dynamic_energy_threshold = True

source = sr.Microphone(sample_rate=16000, device_index=1)

record_timeout = 0.75
phrase_timeout = 3

temp_file = NamedTemporaryFile().name
transcription = ['']

processing = False

phrase_complete = True
noise_tag = False
last_text = None
counter = 0

##with open("rec_text.txt", 'w') as f:
#    f.write("")

with source:
    recorder.adjust_for_ambient_noise(source)

def record_callback(_, audio:sr.AudioData) -> None:
    """
    Threaded callback function to recieve audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)


def ask_gpt(prompt: str, chat_history: list, system_message: str):
    openai.api_key = "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"

    user_prompt = {"role": "user", "content": prompt}
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            *chat_history,
            user_prompt,
        ],
    )

    content = response["choices"][0]["message"]["content"]
    chat_history.append(user_prompt)
    chat_history.append({"role": "assistant", "content": content})

    # Print the text in a green color.
    print("\033[92m" + content + "\033[0m")
    return content


recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)


print("Listening...")
a = 0
chat_history = []
prompt = "1"
    
while True:

    try:
        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty() and not noise_tag:
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            
            # This is the last time we received new audio data from the queue.
            phrase_time = now       

            # Concatenate our current audio data with the latest audio data.
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            wav_data = io.BytesIO(audio_data.get_wav_data())

            # Write wav data to the temporary file as bytes.
            with open(temp_file, 'w+b') as f:
                f.write(wav_data.read())

            # Read the transcription.
            try:
                result = recorder.recognize_google(audio_data)
                text = result.strip()
                if text == last_text:
                    counter += 1
                if counter == 2:
                    noise_tag = True

                last_text = text
            except:
                text = ""

            # If we detected a pause between recordings, add a new item to our transcripion.
            # Otherwise edit the existing one.
            if phrase_complete:
                transcription.append(text)
            else:
                transcription[-1] = text

            # Clear the console to reprint the updated transcription.
            # os.system('cls' if os.name=='nt' else 'clear')
            # for line in transcription:
            #    print(line)

            # Flush stdout.
            print('', end='', flush=True)

            #sleep(0.1)

        else:
            if noise_tag:
                last_sample = bytes()
                phrase_complete = True
                #with open("rec_text.txt", 'a') as f:
                #    f.write(text+"\n")
                print("Input :", text)
                output = ask_gpt(text, chat_history, SYSTEM_MESSAGE)
                print("Output :", output)
                print()
                print("Listening...")
                with data_queue.mutex:
                    data_queue.queue.clear()
                noise_tag = False
                counter = 0
                
            else:
                try :
                    if (not phrase_complete) and now - phrase_time > timedelta(seconds=phrase_timeout):
                        last_sample = bytes()
                        phrase_complete = True
                        #with open("rec_text.txt", 'a') as f:
                        #    f.write(text+"\n")
                        print("Input :", text)
                        output = ask_gpt(text, chat_history, SYSTEM_MESSAGE)
                        print("Output :", output)
                        print()
                        print("Listening...")
                        with data_queue.mutex:
                            data_queue.queue.clear()
                except:
                    pass


    except KeyboardInterrupt:
        break
