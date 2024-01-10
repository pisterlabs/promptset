import io
import os
from tempfile import NamedTemporaryFile

import openai
import speech_recognition as sr
import whisper
from dotenv import load_dotenv
from gtts import gTTS

# Set the OpenAI API key
load_dotenv()
openai.api_key = os.environ["API_KEY"]

# Declare the model and audio model variables
model = "medium.en"
audio_model = whisper.load_model(model)


def get_text():
    """Get user text from microphone."""
    # Create a Recognizer object
    r = sr.Recognizer()

    # Use the Recognizer to listen for audio input
    with sr.Microphone() as source:
        audio = r.listen(source)
        wav_data = io.BytesIO(audio.get_wav_data())

    # Write the wav data to the temporary file as bytes
    temp_file = NamedTemporaryFile().name
    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())

    try:
        # Use the audio model to transcribe the audio data
        result = audio_model.transcribe(temp_file, fp16=False)
        return result['text'].strip()

    # Catch any exceptions
    except Exception as e:
        print("Error: " + str(e))


def play_audio(text):
    """Create a gTTS object and play the audio."""
    # Create a gTTS object with the given text
    tts = gTTS(text=text, lang="en")

    # Create a temporary file to store the audio data
    f = NamedTemporaryFile(delete=True)
    print("Saving audio file to: " + f.name)
    tts.save(f.name)

    # Play the audio file
    print("Playing audio file")
    os.system("ls -l " + f.name)
    os.system("afplay " + f.name)

    # Close the file when we're done with it
    f.close()


def main():
    # Create a Recognizer object
    r = sr.Recognizer()

    # Maintain the conversation context
    conversation = []

    # Keep prompting the user to say something until the user hits 'q'
    while True:
        # prompt the user to speak
        print("Say something!")

        # listen for the user's input, if the text is None, keep trying indefinitely
        text = None
        while text is None:
            text = get_text()
            if text is None:
                print("Didn't understand, try again")

        # Check if the user wants to exit
        user_console_input = input(f"You said: '{text},  press q to quit, any other key to continue: ")
        if user_console_input == "q":
            break

        # Append the user input to the conversation context
        conversation.append(text)

        # print the text that we are sending to OpenAI
        print("Sending to OpenAI: " + text)

        # Send the user input and conversation context to GPT and get the
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="\n".join(conversation),
            max_tokens=1024,
            n=1,
            temperature=0.5,
        )

        # if the response text is empty, skip this iteration
        if response["choices"][0]["text"] == "":
            print("No response from OpenAI, skipping")
            continue

        print("GPT said: " + response["choices"][0]["text"])

        # Play the GPT response
        play_audio(response["choices"][0]["text"])

        # Append the GPT response to the conversation context
        conversation.append(response["choices"][0]["text"])


if __name__ == "__main__":
    main()
