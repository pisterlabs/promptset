import speech_recognition as sr
from gtts import gTTS
import os
import openai

# Set your OpenAI GPT API key
openai.api_key = "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"

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=15
    )
    return response.choices[0].text.strip()

def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("start response.mp3")

def main():
    assistant_name = "eva"
    assistant_description = "I am your virtual assistant created by ukf college students."
    print(f"Hello! I am {assistant_name}. {assistant_description}")

    while True:
        user_input = recognize_speech()

        if user_input:
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            prompt = f"{assistant_name}: {assistant_description}\nUser: {user_input}\n{assistant_name}:"
            response = generate_response(prompt)
            print(f"{assistant_name}: {response}")
            speak(response)

if __name__ == "__main__":
    main()
