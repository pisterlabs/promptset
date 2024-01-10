import speech_recognition as sr
from langdetect import detect 

# Initialize recognizer class (for recognizing the speech)
# r = sr.Recognizer()

# from os import path
# AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "Recording .wav")
# AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
# AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")

use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file
    
# Reading Microphone as source
# listening the speech and store in audio_text variable
speech_file = "Recording.wav"
with open(speech_file, "rb") as audio_file:
    content = audio_file.read()


with sr.Microphone() as source:
    print("Talk")
    audio_text = r.listen(source, timeout=5, phrase_time_limit=10)
    print(type(audio_text))
    print("Time over, thanks")
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    
    try:
        # using google speech recognition
        text = r.recognize_google(audio)
        print("Text: "+ text)
        print("Language: "+ detect(text))
    except Exception as e:
         print("Sorry, I did not get that") 
         print(e)
         
         

from googletrans import Translator, constants
from pprint import pprint
# translate a spanish text to arabic for instance
# init the Google API translator
translator = Translator()
translation = translator.translate(text, dest="en")
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
pprint(translation)



translation1 = translator.translate("are you ok are you not ok my name", dest="hi")
# print(f"{translation1.origin} ({translation1.src}) --> {translation1.text} ({translation1.dest})")
# pprint(translation1)
# print((f"{translation1.origin} ({translation1.src}) --> {translation1.text} ({translation1.dest})").encode("utf-8", "replace").decode("utf-8"))

pprint((f"{translation1.origin} ({translation1.src}) --> {translation1.text} ({translation1.dest})").encode("utf-8", "replace").decode("utf-8"))
pprint(translation1)


from gtts import gTTS

def text_to_speech(text, lang, filename):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

text_to_speech("Hello  i am shreyash", 'en', 'english.mp3')
text_to_speech('Hola, ¿cómo estás?', 'es', 'spanish.mp3')
text_to_speech('Bonjour, comment ça va?', 'fr', 'french.mp3')

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a new chatbot named Charlie
chatbot = ChatBot('Charlie')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Now train the chatbot with the English greetings corpus
trainer.train("chatterbot.corpus.english.greetings")

# Get a response to the input text 'Hello, how are you?'
response = chatbot.get_response('Hello, how are you?')

print(response)

import openai
import os
openai.api_key = "org-lSM2FGW3BNpSIMozUeR8R8IJ"

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
  
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)
text = '''[Your Company Name] - Terms and Conditions

1. Acceptance of Terms:
By accessing or using [Your Company Name]'s products and services, you agree to be bound by these Terms and Conditions.

2. Product Information:
[Your Company Name] reserves the right to modify, update, or discontinue products without prior notice. Product descriptions and specifications are subject to change.

3. Ordering and Payment:
a. All orders are subject to acceptance by [Your Company Name].
b. Prices are in [currency] and are subject to change without notice.
c. Payment must be made in full before the product is shipped.

4. Shipping and Delivery:
a. [Your Company Name] will make reasonable efforts to ensure timely delivery but is not responsible for delays beyond its control.
b. Risk of loss or damage passes to the customer upon delivery.

5. Warranty:
a. [Your Company Name] provides a limited warranty on its products. Please refer to the warranty statement for details.
b. The warranty is void if the product has been tampered with, modified, or repaired by unauthorized personnel.

6. Returns and Refunds:
a. Customers may return products within a specified period for a refund, subject to [Your Company Name]'s return policy.
b. Refunds will be issued in the original form of payment.

7. Intellectual Property:
a. All intellectual property rights related to [Your Company Name]'s products are owned by [Your Company Name].
b. Users are prohibited from reproducing, distributing, or using any content without explicit permission.

8. Limitation of Liability:
[Your Company Name] is not liable for any indirect, incidental, special, or consequential damages arising out of or in connection with the use of its products.

9. Governing Law:
These Terms and Conditions are governed by the laws of [Your Jurisdiction].

10. Modification of Terms:
[Your Company Name] reserves the right to update or modify these Terms and Conditions at any time without prior notice.

Contact Information:
[Your Company Name]
[Address]
[Email]
[Phone]'''

import os

# Set environment variable
os.environ['REPLICATE_API_TOKEN'] = 'r8_534R8BS65dTbyBgPvaBMU2kwAqOr6VY2cFEx3'
# Access the environment variable
value = os.environ.get('REPLICATE_API_TOKEN')
print(value)
import os

REPLICATE_API_TOKEN = 'r8_534R8BS65dTbyBgPvaBMU2kwAqOr6VY2cFEx3'
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

import replicate
output = replicate.run(
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
    input={"prompt": "Summarize the terms and condition which is given below" + text, "max_length": 10000}
)
# The replicate/vicuna-13b model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
for item in output:
    # https://replicate.com/replicate/vicuna-13b/api#output-schema
    print(item, end="")