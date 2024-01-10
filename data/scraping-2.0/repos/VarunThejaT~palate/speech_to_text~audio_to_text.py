from pydub import AudioSegment
import math
import openai
import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

from nr_openai_observability import monitor

monitor.initialization()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Set up the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")

file_location = "/home/varun/Downloads/TheTimFerrissShow_Eric Cressey.mp3"
file_stats = os.stat(file_location)

print(file_stats)
print(f'File Size in Bytes is {file_stats.st_size}')
sound = AudioSegment.from_mp3(file_location)

MBBytes=2**20
num_slices=math.ceil(file_stats.st_size/MBBytes/25)
print(f"num slices: {num_slices}")

#whisper
transcription_arr = []
slice_len   = len(sound) / num_slices
for i in range(num_slices):
  if i==num_slices-1:
    new = sound[i*slice_len:]
  else:
    new = sound[i*slice_len:(i+1)*slice_len]

  # writing mp3 files is a one liner
  new.export("file_"+ str(i) +".mp3", format="mp3")

  file = open("file_"+ str(i) +".mp3", "rb")
  print("calling whisper API")
  part_transcription = openai.Audio.transcribe("whisper-1", file)
  transcription_arr.append(part_transcription.text)

#Full text
full_transcription = "".join(transcription_arr)
words_of_transcription = full_transcription.split(" ")

GPT_slices=math.ceil(num_tokens_from_string(str(full_transcription), "cl100k_base")/3000)

#summarize full text
summarization=[]
for i in range(GPT_slices):
  messages = [ {"role": "system", "content": 
                "You are a summarization machine for a podcast."} ]

  message="Convert the text given by ```" +  full_transcription[i*len(full_transcription)//GPT_slices:(i+1)*len(full_transcription)//GPT_slices] + " ``` into a " + str(2000//GPT_slices) + " word text" 

  if message:
      messages.append(
          {"role": "user", "content": message},
      )
      print("Calling chatgpt")
      chat = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", messages=messages
      )
  reply = chat.choices[0].message.content
  print("Number of words:" + str(len(reply.split())))
  print(f"ChatGPT: {reply}")
  messages.append({"role": "assistant", "content": reply})
  summarization.append(reply)


summary = "".join(summarization)
words_of_summary = summary.split(" ")

mins = 5

summarization=[]
messages = [ {"role": "system", "content": 
              "You are a text to podcast machine."} ]


message="Can you give me a text that is written in a way that David Attenborough would read it? Start with a catch phrase and mention 'Palate' as Sponsor. Make the podcast "+ str(mins*132) +" words long. \\ Content: ```" + summary + "```"   

if message:
    messages.append(
        {"role": "user", "content": message},
    )
    print("Calling chatgpt")
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
reply = chat.choices[0].message.content
print("Number of words:" + str(len(reply.split())))
print(f"ChatGPT: {reply}")

#write to text
with open('TheTimFerrissShow_Eric_Cressey_'+ str(mins) +'_mins.txt', 'w') as f:
    f.write(reply)
