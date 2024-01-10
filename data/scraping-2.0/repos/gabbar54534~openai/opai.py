import openai
import pyttsx3
import speech_recognition as sr
from gtts import gTTS

openai.api_key = "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"
engine = pyttsx3.init()


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print('Speech Recognition could not understand audio')
    except sr.RequestError:
        print('Could not request results from Speech Recognition service')
    return None


def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        print("Say 'genius' to start")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "genius":
                    filename = "input.wav"
                    print("Say your question")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())
                    text = transcribe_audio_to_text(filename)
                    if text:
                        print(f"You said: {text}")
                        response = generate_response(text)
                        print(f"GPT-3 says: {response}")
                        tts = gTTS(text=response, lang='en')
                        tts.save("sample.mp3")
                        speak_text(response)
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
            except sr.RequestError:
                print("Could not request results from Speech Recognition service")
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
