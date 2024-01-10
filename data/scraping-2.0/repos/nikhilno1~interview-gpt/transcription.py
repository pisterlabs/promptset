import os
import datetime
import openai
import streamlit as st

from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment


# import API key from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")

openai.api_key = os.getenv("OPENAI_API_KEY", "")
if openai.api_key == "":
    openai.api_key = st.secrets["OPENAI_API_KEY"]

def transcribe(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
    return transcript

def remove_files_with_prefix(directory_path, prefix):
    try:
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Iterate through all the items in the directory
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                
                # Check if the item is a file and starts with the specified prefix
                if os.path.isfile(item_path) and item.startswith(prefix):
                    # Remove the file
                    os.remove(item_path)
        else:
            print(f"The directory '{directory_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def save_audio_file(audio_bytes, file_extension, recordings_folder):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{recordings_folder}/audio_{timestamp}.{file_extension}"
    #print("recording path", file_name)

    remove_files_with_prefix(recordings_folder, "audio_")

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    # Reduce the bitrate since Whisper API has a 25 MB file size limit that gets exceeded with just 2 min 30 sec audio
    audio = AudioSegment.from_file(file_name)
    name, extension = file_name.rsplit(".", 1)
    new_file_name = f"{name}_reduced.{extension}"
    audio.export(new_file_name, bitrate='128k', format='mp3')  
    os.remove(file_name)  

    return new_file_name


def transcribe_audio(file_path, transcripts_folder):
    """
    Transcribe the audio file at the specified path.

    :param file_path: The path of the audio file to transcribe
    :return: The transcribed text
    """
    remove_files_with_prefix(transcripts_folder, "ts_")

    with open(file_path, "rb") as audio_file:
        transcript = transcribe(audio_file)

    return transcript["text"]


def run_transcription_app(recordings_folder):
    """
    Main function to run the Transcription app.
    """
    st.markdown("---")
    st.header("Record your answer")
    st.markdown('<small>Click the mic to start/stop the recording. Then click Analyze. <i>(Max 4 mins.)</i></small>', unsafe_allow_html=True)    
   
    # Record Audio tab
    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0),  pause_threshold=240.0)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        save_audio_file(audio_bytes, "mp3", recordings_folder)
        st.session_state.analyze_button_disable = False            


def do_transcribe(recordings_folder, transcripts_folder):
    # Find the newest audio file
    subdirectory_path = os.path.join(".", recordings_folder)    

    audio_file_path = max(
        [os.path.join(subdirectory_path, f) for f in os.listdir(subdirectory_path) if f.startswith("audio")],
        key=os.path.getctime,
    )

    # Transcribe the audio file
    transcript_text = transcribe_audio(audio_file_path, transcripts_folder)

    # Display the transcript
    st.header("Transcript")
    st.write(transcript_text)
    
    # Save the transcript to a text file    
    a_file = os.path.splitext(os.path.basename(audio_file_path))[0]
    with open(transcripts_folder + "/ts_" + a_file + ".txt", "w") as f:
        f.write(transcript_text)
        
    return transcript_text
    # Provide a download button for the transcript
    #st.download_button("Download Transcript", transcript_text)
   
