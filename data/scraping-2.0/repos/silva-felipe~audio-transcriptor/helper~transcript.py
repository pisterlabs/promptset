import os
from dotenv import load_dotenv
import openai
from google.cloud import secretmanager
from pydub import AudioSegment
import streamlit as st

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Specify the directory you want to ensure exists
directory = '../transcripts'

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

def access_secret_version(project_id, secret_id, version_id):
    """
    Access the payload of a given secret version.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')

# Use the function
openai.api_key = access_secret_version("audio-transcriptor-392502", "OPENAI_API_KEY", "latest")

def split_audio(filename, chunk_length_ms=120000):  # default chunk length is 60 seconds
    audio = AudioSegment.from_file(filename)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunks.append(audio[i:i+chunk_length_ms])
    return chunks

@st.cache_data
def transcript(filename):
    base_filename, _ = os.path.splitext(os.path.basename(filename))
    chunks = split_audio(filename)
    transcripts = []
    for i, chunk in enumerate(chunks):
        chunk_filename = f"./transcripts/{base_filename}_chunk{i}.wav"
        chunk.export(chunk_filename, format="wav")  # save chunk to a temporary file
        with open(chunk_filename, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
            transcripts.append(response['text'])
        os.remove(chunk_filename)  # delete the temporary file

    # join all transcripts into a single string
    transcripts = " ".join(transcripts)

    with open(f"./transcripts/{base_filename}.txt", "w") as f:
        f.write(transcripts)
    
    return transcripts

# print(transcript("./downloads/IA CÂMERA 3D e INFRAVERMELHO conheça o FREE FLOW.mp4"))