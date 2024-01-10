import openai
import speech_recognition as sr
import pyttsx3
import subprocess
import streamlit as st

openai.api_key = "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"

dataset = {
"directory ": "dir.py",
"who am I": "whoami.py",
"wifi off": "wifi.py",
"Browser": "brave.py",
"shutdown": "shutdown.py",
"notepad": "notepad.py"
}

r = sr.Recognizer()


def listen_and_convert_to_text():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio_data = r.record(source, duration=5)  # record for 5 seconds
    recognized_text = r.recognize_google(audio_data)
    return recognized_text

def execute_command(prompt):
    st.write(prompt)
    if prompt in dataset:
        response = dataset[prompt]
        result = subprocess.run(["python", response], stdout=subprocess.PIPE)
        response2 = result.stdout.decode("utf-8").strip()
        st.write(response2)
        engine = pyttsx3.init()
        engine.say(response2)
        engine.runAndWait()
    else:
        completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
        )
        message = completions.choices[0].text
        st.write(message)
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()

def main():
    st.title("Voice Controlled Web App")
    st.write("Press the button to listen ")
    st.write("Commands Available to run:\n1.Directory\n2.Whoami\n3.Wifioff\n4.Browser\n5.Shutdown\n6.Notepad")
    if st.button("Listen"):
        prompt = listen_and_convert_to_text()
        execute_command(prompt)

if __name__ == "__main__":
    main()
