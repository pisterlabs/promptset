# Import necessary libraries
import gradio as gr
import openai, config, subprocess
from pydub import AudioSegment

# Set OpenAI API key from config
openai.api_key = config.OPENAI_API_KEY

# Initialize a list of messages with a system message
messages = [{"role": "system", "content": 'You are a therapist. Respond to all input in 25 words or less, keep it interesting.'}]

# Define a function to transcribe audio input and generate a response
def transcribe(audio):
    global messages

    # Convert the input audio to a WAV format
    audio_file = AudioSegment.from_file(audio)
    audio_file.export("converted_audio.wav", format="wav")
    
    # Transcribe the converted audio file
    with open("converted_audio.wav", "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
        print("Transcript:", transcript["text"])

    # Append the transcribed message to the messages list
    messages.append({"role": "user", "content": transcript["text"]})

    # Generate a response using OpenAI ChatCompletion
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    print("Response:", response)

    # Extract the system message from the response
    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    # Convert the system message to speech
    subprocess.call(["say", system_message['content']])

    # Compile a chat transcript excluding system messages
    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

# Create a Gradio interface with the transcribe function, using audio input and text output
iface = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text")

# Launch the Gradio interface
iface.launch()
