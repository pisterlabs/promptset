import math
import requests
import tiktoken
from newspaper import fulltext
import openai
from dotenv import load_dotenv

load_dotenv()

# Set up the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")

html = requests.get("https://www.brookings.edu/research/how-artificial-intelligence-is-transforming-the-world/").text
text=fulltext(html)
full_transcription = text
print(full_transcription)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

GPT_slices = math.ceil(num_tokens_from_string(str(text), "cl100k_base")/3000)
GPT_slices

summarization=[]
for i in range(GPT_slices):
  messages = [ {"role": "system", "content": 
                "You are a summarization machine for a podcast."} ]


  message = "Convert the text given by ```" +  text[i*len(text)//GPT_slices:(i+1)*len(full_transcription)//GPT_slices] + " ``` into a " + str(2000//GPT_slices) + " word text" 

  if message:
      messages.append(
          {"role": "user", "content": message},
      )
      chat = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", messages=messages
      )
  reply = chat.choices[0].message.content
  messages.append({"role": "assistant", "content": reply})
  summarization.append(reply)

summary = "".join(summarization)
words_of_summary = summary.split(" ")


mins=3

messages = [ {"role": "system", "content": 
              "You are a text to podcast machine."} ]


message="Can you give me a text that is written in a way that David Attenborough would read it? Start with a catch phrase and mention 'Palate' as Sponsor. Make the podcast "+ str(mins*132) +" words long. \\ Content: ```" + summary + "```"   

if message:
    messages.append(
        {"role": "user", "content": message},
    )
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
reply = chat.choices[0].message.content
print("Number of words:" + str(len(reply.split())))
print(f"ChatGPT: {reply}")
messages.append({"role": "assistant", "content": reply})


with open('david_AI_article_'+ str(mins) +'_mins.txt', 'w') as f:
    f.write(reply)