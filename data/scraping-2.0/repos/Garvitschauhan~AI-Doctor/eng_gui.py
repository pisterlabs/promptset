import openai
import pyttsx3
import tkinter as tk
import speech_recognition as sr
import threading
import math

# Initialize the text-to-speech engine
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Initialize the Tkinter window
window = tk.Tk()
window.title("MED BAY")
window.geometry("800x600")
window.configure(bg="black")  # Set the background color to black

# Set the OpenAI API key
openai.api_key = "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"

# Function to add text with fade-out animation
def add_to_conversation(text, role="AI Bot"):
    conversation_text.config(state=tk.NORMAL)
    conversation_text.insert(tk.END, f"{role}: {text}\n")
    conversation_text.config(state=tk.DISABLED)
    conversation_text.see(tk.END)

    fade_out(conversation_text, len(text) + len(role) + 2)  # Fade-out the text

def fade_out(widget, delay):
    widget.after(delay, lambda: fade(widget, 0))

def fade(widget, alpha):
    alpha -= 0.01
    widget.configure(insertbackground='white')
    widget.configure(insertbackground='white')
    if alpha > 0:
        widget.after(50, lambda: fade(widget, alpha))
    else:
        widget.configure(insertbackground='black')
        widget.delete("1.0", "2.0")  # Remove the first line of text
        widget.see("end")  # Scroll to the end

# Function to speak the text
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Create a black label and an entry for user input
input_label = tk.Label(window, text="Your Message:", font=("Arial", 12), bg="black", fg="white")
input_label.pack(pady=10)

input_entry = tk.Entry(window, width=50, font=("Arial", 12), bg="black", fg="white")
input_entry.pack(pady=10)

# Function to process user input and get a response
def process_input():
    user_input = input_entry.get()
    if user_input.strip():
        add_to_conversation(user_input, role="You")
        conversation.append({"role": "user", "content": user_input})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=conversation, temperature=0, max_tokens=3000
            )
            bot_reply = response["choices"][0]["message"]["content"]
            add_to_conversation(bot_reply)
            conversation.append({"role": "assistant", "content": bot_reply})
            speak_text(bot_reply)
        except Exception as e:
            error_message = "An error occurred: " + str(e)
            add_to_conversation(error_message)

    input_entry.delete(0, tk.END)

send_button = tk.Button(window, text="Send", command=process_input, font=("Arial", 12), bg="black", fg="white")
send_button.pack()

# Function to process voice input
def process_voice_input():
    def voice_thread():
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source)
                voice_input = recognizer.recognize_google(audio)
                if voice_input.strip():
                    add_to_conversation(voice_input, role="You")
                    conversation.append({"role": "user", "content": voice_input})

                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo", messages=conversation, temperature=0, max_tokens=3000
                        )
                        bot_reply = response["choices"][0]["message"]["content"]
                        add_to_conversation(bot_reply)
                        conversation.append({"role": "assistant", "content": bot_reply})
                        speak_text(bot_reply)
                    except Exception as e:
                        error_message = "An error occurred: " + str(e)
                        add_to_conversation(error_message)
            except sr.UnknownValueError:
                error_message = "Speech Recognition could not understand the audio"
                add_to_conversation(error_message)
            except sr.RequestError as e:
                error_message = "Speech Recognition error: " + str(e)
                add_to_conversation(error_message)
            except Exception as e:
                error_message = "An error occurred: " + str(e)
                add_to_conversation(error_message)

    threading.Thread(target=voice_thread).start()

# Create a black button for voice input
voice_button = tk.Button(window, text="Voice Input", command=process_voice_input, font=("Arial", 12), bg="black", fg="white")
voice_button.pack()

# Create a text box to display the conversation
conversation_text = tk.Text(window, wrap=tk.WORD, font=("Arial", 12), bg="black", fg="white")
conversation_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Initialize the conversation
conversation = []

# Start the Tkinter main loop
window.mainloop()