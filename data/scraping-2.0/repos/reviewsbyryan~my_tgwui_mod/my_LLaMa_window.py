#------------------ Description ------------------
# This is a simple chat window that uses the text gen web ui API to generate responses.
# 
# 
#------------------ Imports ------------------
import time
import speech_tool as st
import speech_recognition as sr

import tkinter as tk
import threading

# ------------------ Model ------------------
import os
os.environ['OPENAI_API_KEY']="sk-111111111111111111111111111111111111111111111111"
os.environ['OPENAI_API_BASE']="http://127.0.0.1:5001/v1"
import openai

def query_model(message):
  response = openai.ChatCompletion.create(
    model="x",
    messages = [
      { 'role': 'system', 'content': "Answer in a consistent style." },
      {'role': 'user', 'content': message},
    ]
  )

  text = response['choices'][0]['message']['content']
  return text

#------------------ Functions ------------------

def get_response(input):

  #lock the buttons so they cannot be clicked
  hear_button.config(state=tk.DISABLED)
  send_button.config(state=tk.DISABLED)

  #get response from model and display it incrementally
  response = query_model(input)
  text_display.config(state=tk.NORMAL)
  text_display.insert(tk.END,"\nAI: ")
  for i, token in enumerate(response):
    text_display.insert(tk.END,token + "")
    text_display.update()
    time.sleep(0.1)
  text_display.config(state=tk.DISABLED)
 
   #lock the buttons so they cannot be clicked
  hear_button.config(state=tk.NORMAL)
  send_button.config(state=tk.NORMAL)

def hear_message():

  # TODO setup listening message event
  # 1. show a message in chat that says "Listening..."
  # 2. listen for a message
  # 3. show a message in chat that says "Processing..."
  # 4. process the message
  # 5. display the message in chat
  # 6. display the response in chat
  # 7. repeat

  message = st.microphone.run()
  threading.Thread(target=get_response(message)).start()

  if message == "":
    message = "MESSAGE NOT HEARD, it is empty input and you should respond by asking #how you may assist because you did not hear me."
  else:
    #get input from microphone and concatenate words into a single string
    text_display.config(state=tk.NORMAL)
    text_display.insert(tk.END, "\nAI: " + message + "\n")
    text_display.config(state=tk.DISABLED)

    #display the response in a new thread using Threading
    threading.Thread(target=get_response(message)).start()
"""
def hear_message():
  r = sr.Recognizer()
  mic = sr.Microphone()

  with mic as source:
    r.adjust_for_ambient_noise(source)

    # show a message in chat that says "Listening..."
    text_display.config(state=tk.NORMAL)
    text_display.insert(tk.END, "\n" + "Listening..." + "\n")
    text_display.config(state=tk.DISABLED)

    print("Say something!")
    audio = r.listen(source)

    try:
        text_display.config(state=tk.NORMAL)
        text_display.insert(tk.END, "\n" + "Processing..." + "\n")
        text_display.config(state=tk.DISABLED)

        words = r.recognize_google(audio)

        print("You said: " + " ".join(words))
    except sr.UnknownValueError:
        print("Could not understand you.")

    try:
      message = "".join(words)
    except:
      message = "MESSAGE NOT HEARD, it is empty input and you should respond by asking #how you may assist because you did not hear me."

    if message == "":
      message = "MESSAGE NOT HEARD, it is empty input and you should respond by asking #how you may assist because you did not hear me."

      #get input from microphone and concatenate words into a single string
      text_display.config(state=tk.NORMAL)
      text_display.insert(tk.END, "\nAI: " + message + "\n")
      text_display.config(state=tk.DISABLED)
  
      #display the response in a new thread using Threading
      threading.Thread(target=get_response(message)).start()
    else:
      #get input from microphone and concatenate words into a single string
      text_display.config(state=tk.NORMAL)
      text_display.insert(tk.END, "\nAI: " + message + "\n")
      text_display.config(state=tk.DISABLED)

      #display the response in a new thread using Threading
      threading.Thread(target=get_response(message)).start()
"""
def send_message():
  #get input from entry and display it
  message = entry.get()

  if message == "":
    return
  else:

    text_display.config(state=tk.NORMAL)
    text_display.insert(tk.END, "\n" + "You: " + message + "\n")
    text_display.config(state=tk.DISABLED)
    entry.delete(0, tk.END)

    #display the response in a new thread using Threading
    threading.Thread(target=get_response(message)).start()
# ------------------ GUI ------------------
root = tk.Tk()
root.title("Chat Window")

# Create a text display area with vertical scrollbar
text_display = tk.Text(root, wrap=tk.WORD)
text_display.config(state=tk.DISABLED)
text_display.pack(fill=tk.BOTH, expand=True)

# Create an input field and send button
entry = tk.Entry(root)
entry.pack(fill=tk.BOTH, expand=True)

# Create a button to hear the message
hear_button = tk.Button(root, text="Hear", command=hear_message)
hear_button.pack(fill=tk.BOTH, expand=True)


send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(fill=tk.BOTH, expand=True)

root.mainloop()
