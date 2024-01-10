import speech_recognition as sr
import openai
# create a recognizer object
r = sr.Recognizer()

# use the microphone as the audio source
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# recognize speech using Google Speech Recognition
try:
    prompt = r.recognize_google(audio)
    print("You said: {}".format(prompt))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
# 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 
openai.api_key = "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"

# prompt = "Hello, how are you?"
response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=50)

print(response.choices[0].text)
