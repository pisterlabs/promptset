# Transcribe audio to text
import speech_recognition as sr
# Convert text to speech
import pyttsx3
# Access GPT API
import openai
# Import os for accessing environment variables
import os

# Initialize API key
openai.api_key = os.environ["API_KEY"]
# Initialize text-to-speech engine
engine = pyttsx3.init()
# Set up assistant's voice
voices = engine.getProperty("voices")
engine.setProperty("voices", voices[0].id)

# Initialize an object of Recognizer class
r = sr.Recognizer()
# Set up microphone
mic = sr.Microphone(device_index=0)

# Initialize conversation variable
conversation = ""
# Initialize user and bot name
user_name = "Matt"
bot_name = "ChatGPT"

while True:
    with mic as source:
        print("\nListening to you...")
        # Fetch the user's audio
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio = r.listen(source)
    print("No longer listening to you")

    try:
        # Convert voice into text
        user_input = r.recognize_google(audio)
    # Catch any exceptions
    except:
        continue

    # Setting up user's prompt to be understood by OpenAI
    prompt = user_name + ":" + user_input + "\n" + "bot_name" + ":"
    # Append user's query to the conversation string
    conversation += prompt

    # Get the input from OpenAI and convert it into speech
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=conversation,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Store the OpenAI's response in a string variable
    response_str = response["choices"][0]["text"].replace("\n", "")
    response_str = response_str.split(
        user_name + ":", 1)[0].split(bot_name + ":", 1)[0]
    # Append OpenAI's response to the conversation string
    conversation += response_str + "\n"

    # Print out the conversation
    print(f"{user_name}: {user_input} ")
    print(f"{bot_name}: {response_str}")

    # Convert OpenAI's response string to voice
    engine.say(response_str)
    engine.runAndWait()
