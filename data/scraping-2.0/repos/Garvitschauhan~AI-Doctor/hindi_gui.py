import openai
import pyttsx3
import tkinter as tk
import speech_recognition as sr
import threading

# टेक्स्ट-टू-स्पीच इंजन को आरंभ करें
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Tkinter विंडो को आरंभ करें
window = tk.Tk()
window.title("MED BAY")
window.geometry("800x600")
window.configure(bg="black")  # पृष्ठभूमि रंग को काला सेट करें

# OpenAI API कुंजी सेट करें
openai.api_key = "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"

# फेड-आउट एनीमेशन के साथ पाठ जोड़ने के लिए फ़ंक्शन
def add_to_conversation(text, role="AI बॉट"):
    conversation_text.config(state=tk.NORMAL)
    conversation_text.insert(tk.END, f"{role}: {text}\n")
    conversation_text.config(state=tk.DISABLED)
    conversation_text.see(tk.END)

    fade_out(conversation_text, len(text) + len(role) + 2)  # पाठ को फड़काना

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
        widget.delete("1.0", "2.0")  # पहली पंक्ति को हटाएं
        widget.see("end")  # अंत में स्क्रॉल करें

# पाठ बोलने के लिए फ़ंक्शन
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# काले लेबल और उपयोगकर्ता इनपुट के लिए एंट्री बनाएं
input_label = tk.Label(window, text="आपका संदेश:", font=("Arial", 12), bg="black", fg="white")
input_label.pack(pady=10)

input_entry = tk.Entry(window, width=50, font=("Arial", 12), bg="black", fg="white")
input_entry.pack(pady=10)

# उपयोगकर्ता इनपुट प्रोसेस करने और प्रतिक्रिया प्राप्त करने के लिए फ़ंक्शन
def process_input():
    user_input = input_entry.get()
    if user_input.strip():
        add_to_conversation(user_input, role="आप")
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
            error_message = "एक त्रुटि हुई: " + str(e)
            add_to_conversation(error_message)

    input_entry.delete(0, tk.END)

send_button = tk.Button(window, text="भेजें", command=process_input, font=("Arial", 12), bg="black", fg="white")
send_button.pack()

# आवाज इनपुट प्रोसेस करने के लिए फ़ंक्शन
def process_voice_input():
    def voice_thread():
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source)
                voice_input = recognizer.recognize_google(audio)
                if voice_input.strip():
                    add_to_conversation(voice_input, role="आप")
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
                        error_message = "एक त्रुटि हुई: " + str(e)
                        add_to_conversation(error_message)
            except sr.UnknownValueError:
                error_message = "आवाज संज्ञान करने में समस्या आई"
                add_to_conversation(error_message)
            except sr.RequestError as e:
                error_message = "आवाज संज्ञान त्रुटि: " + str(e)
                add_to_conversation(error_message)
            except Exception as e:
                error_message = "एक त्रुटि हुई: " + str(e)
                add_to_conversation(error_message)

    threading.Thread(target=voice_thread).start()

# काला बटन बनाएं आवाज इनपुट के लिए
voice_button = tk.Button(window, text="आवाज इनपुट", command=process_voice_input, font=("Arial", 12), bg="black", fg="white")
voice_button.pack()

# वार्ता प्रदर्शित करने के लिए एक पाठ बॉक्स बनाएं
conversation_text = tk.Text(window, wrap=tk.WORD, font=("Arial", 12), bg="black", fg="white")
conversation_text.pack(pady=10)

# Initialize the conversation
conversation = []

# Start the Tkinter main loop
window.mainloop()
