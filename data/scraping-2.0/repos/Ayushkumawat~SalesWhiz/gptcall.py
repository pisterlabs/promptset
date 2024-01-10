import openai
import speech_recognition as sr
from gtts import gTTS
import playsound
import os

# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
openai.api_key = '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'
messages = []
messages.append({"role": "system", "content": "You are Mahindra's AI sales executive which talks over calls and resolve quries about mahindra vehicles, features, and even ask to book a test drive, talk in short one on one conversation, the responses should not be more than 5 lines and should be only related to mahindra only also give the ending conversation if no questions are left"})

def chat_with_ai(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    ai_response = response["choices"][0]["message"]["content"]

    # Check if the user provided their name or address in the input
    if "name" in user_input:
        user_name = user_input  # Store the name in the user_name variable
    elif "address" in user_input:
        user_address = user_input

    messages.append({"role": "assistant", "content": ai_response})
    return ai_response

def text_to_speech(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    playsound.playsound("response.mp3")
    os.unlink(r"response.mp3")
def main():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak something...")
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        print("You said:", user_input)

        ai_response = chat_with_ai(user_input)
        print("AI Response:", ai_response)

        text_to_speech(ai_response)

    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
    except sr.RequestError as e:
        print("Error:", e)
while True:
    main()
