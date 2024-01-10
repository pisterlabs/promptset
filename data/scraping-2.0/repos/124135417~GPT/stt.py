import speech_recognition as sr
import openai

# Initialize recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Say something!")
    audio = recognizer.listen(source)

# Transcribe speech to text
try:
    speech_text = recognizer.recognize_google(audio)
    print("Transcribed text: " + speech_text)
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Error: {0}".format(e))

openai.api_key = 'your-api-key-here'

# Send text to GPT and get a response
response = openai.Completion.create(
  engine="gpt-4-1106-preview",  # or another model
  prompt=speech_text,
  max_tokens=50  # Adjust based on your needs
)

gpt_response = response.choices[0].text.strip()
print("GPT response: ", gpt_response)
