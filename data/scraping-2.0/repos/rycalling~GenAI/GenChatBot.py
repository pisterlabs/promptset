import spacy
import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import numpy as np
import noisereduce as nr
import pyttsx3
import openai
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.chat.util import Chat, reflections

openai.api_key= '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'
model_id = "gpt-3.5-turbo"
pairs = [
			[
				r"(.*)my name is (.*)",
				["Hello %2, How are you today ?",] 
			],
			[
				r"(.*)help(.*)",
				["I can help you ",]
			],
			[
				r"(.*) your name ?",
				["My name is Gen AI, but you can just call me anything and I'm your assistant for the day .",]
			],
				[
				r"(.*)fraud case(.*)",
				["Please share the reference or case numbr received in your email",]
			],
				[
				r"(.*)(crypto|gambling|3dsecure|)(.*)",
				["can you please confirm the transaction amount",]
			],
				[
				r"(.*)debit card fraud(.*)",
				["Please share the reference or case numbr received in your email",]
			],
			[
				r"how are you (.*) ?",
				["I'm doing very well", "i am great !"]
			],
			[
				r"sorry (.*)",
				["Its alright","Its OK, never mind that",]
			],
			[
				r"i'm (.*) (good|well|okay|ok)",
				["Nice to hear that","Alright, great !",]
			],
			[
				r"(hi|hey|hello|hola|holla)(.*)",
				["Hello", "Hey there",]
			],
			[
				r"(.*)created(.*)",
				["Natwest Group created me ","top secret ;)",]
			],
			[
				r"quit",
				["Bye for now. See you soon :) ","It was nice talking to you. See you soon :)"]
			],
			[
				r"(.*)",
				['That is nice to hear']
			],
		]

reflections = {"i am": "you are", 
			   "i was": "you were", 
			   "i": "you", 
			   "i'm": "you are", 
			   "i’d": "you would", 
			   "i’ve": "you have", 
			   "i’ll": "you will", 
			   "my": "your", 
			   "you are": "I am", 
			   "you were": "I was", 
			   "you’ve": "I have", 
			   "you’ll": "I will", 
			   "your": "my", 
			   "yours": "mine", 
			   "you": "me", 
			   "me": "you"
			  }

chat = Chat(pairs, reflections)

class NLPChatbotUI:
  def __init__(self, root):
    self.root = root
    self.root.title("GEN AI Chatbot")

    self.chat_area = Text(root, wrap=tk.WORD, state=tk.DISABLED)
    self.scrollbar = Scrollbar(root, command=self.chat_area.yview)
    self.chat_area.config(yscrollcommand=self.scrollbar.set)

    self.user_input = Entry(root)


    self.voice_button = Button(root, text="Voice", command=self.voice_input)

    self.chat_area.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.user_input.pack(padx=10, pady=5, expand=True, fill=tk.X)

    self.voice_button.pack(pady=5)

    self.nlp = spacy.load("en_core_web_sm")
    self.add_bot_message("Natwest Agent: Hi! How can I help you?")
    self.recognizer = sr.Recognizer()

  def voice_input(self):
    while True:
        try:
          self.recognizer = sr.Recognizer()

          with sr.Microphone() as source:
            self.chat_area.update_idletasks()
            self.recognizer.adjust_for_ambient_noise(source)
			greet_msg = self.get_gpt_response("Consider yourself as Gen Ai, who is helping bank customers. Greet the customer who has just called in")
            self.text_to_speech(greet_msg)
			print("Natwest Agent: ", greet_msg)
            print("Please speak something...")
            audio = self.recognizer.listen(source)

              # Convert audio to NumPy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)

            # Reduce noise from audio
            reduced_noise = nr.reduce_noise(y=audio_data, sr=audio.sample_rate)

            # Convert the reduced noise audio back to AudioData
            reduced_noise_audio = sr.AudioData(
                reduced_noise.tobytes(),
                sample_rate=audio.sample_rate,
                sample_width=reduced_noise.dtype.itemsize,
            )

            recognized_text = self.recognizer.recognize_google(reduced_noise_audio)
            self.add_user_message("Customer: " + recognized_text)
            response = self.process_message(recognized_text)
            self.add_bot_message("Natwest Agent: " + response)

            self.text_to_speech(response)

            print("Recognized text:", recognized_text)
			
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))

  def process_message(self, user_input):
    user_input = user_input.lower()
	threshold_ratio = 60
	
    if fuzz.token_sort_ratio(user_input, "hello") >= threshold_ratio:
      return "Sorry, I could not recognise the linked account to this number, can you please confirm your bank account number"
    if fuzz.token_sort_ratio(user_input, "hey") >= threshold_ratio:
      return "Sorry, I could not recognise the linked account to this number, can you please confirm your bank account number"
    elif fuzz.token_sort_ratio(user_input, "how are you") >= threshold_ratio:
      return "I'm your assistant for the day and here to help."
    elif fuzz.token_sort_ratio(user_input, "my account number is one two three four") >= threshold_ratio:
      return "thank you for confirming the bank account, can you please confirm your name"
    elif fuzz.token_sort_ratio(user_input, "my name is") >= threshold_ratio:
      return "Great, thank you Stalin. I can see there is a Fraud case created in your account. Would you like to know its status"
    elif fuzz.token_sort_ratio(user_input, "yes") >= threshold_ratio:
      return "can you please confirm the case number received by email"
    elif fuzz.token_sort_ratio(user_input, "my case reference number is three four five") >= threshold_ratio:
      return "Thanks for confirming. your case is under progress. we would like a bit more information to progress it further"
    elif fuzz.token_sort_ratio(user_input, "sure") >= threshold_ratio:
      return "What was the purpose of the transaction"
     elif fuzz.token_sort_ratio(user_input, "Buying Crypto") >= threshold_ratio:
      return "can you please confirm the transaction amount"
    elif fuzz.token_sort_ratio(user_input, "five pounds GBP") >= threshold_ratio:
      return "Thanks for confirming. can you please provide the details of the retailer"
    elif fuzz.token_sort_ratio(user_input, "yes it was Binance amazon") >= threshold_ratio:
      return random.choice(["thank you for sharing the details, we will move the case to our investigations team for review the details and progress the case","thank you for sharing the details, we will progress the case for a refund"])
    else:
      return chat.respond(user_input)

  def add_user_message(self, message):
    self.chat_area.config(state=tk.NORMAL)
    self.chat_area.insert(tk.END, message + "\n")
    self.chat_area.config(state=tk.DISABLED)
    self.chat_area.see(tk.END)

  def add_bot_message(self, message):
    self.chat_area.config(state=tk.NORMAL)
    self.chat_area.insert(tk.END, message + "\n", "bot")
    self.chat_area.config(state=tk.DISABLED)
    self.chat_area.see(tk.END)

  def get_gpt_response(self, input_msg):
    try:
      gptChat = openai.ChatCompletion.create(
      model=model_id,
      messages=[{"role": "user", "content": input_msg}])
      resp = gptChat.choices[0].message.content
      return resp
    except Exception as e:
      print(f"Error: {e}")


  def text_to_speech1(self, text, output_file="output.mp3", lang="en"):
    try:
      tts = gTTS(text=text, lang=lang)
      tts.save(output_file)
      print(f"Text saved as '{output_file}'")
      os.system(f"start {output_file}")  # This plays the generated audio on Windows
    except Exception as e:
      print(f"Error: {e}")

  def text_to_speech(self, text):
    try:
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Convert text to speech
        engine.say(text)
        engine.runAndWait()

        print("Text converted to speech successfully.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
  root = tk.Tk()
  chatbot_ui = NLPChatbotUI(root)
  root.mainloop()