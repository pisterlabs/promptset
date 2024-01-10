
import streamlit as st
import openai
from pydub import AudioSegment
import os
import tempfile

st.title("Audio to :blue[_Summary_] by small chunks")

st.text("use the OpenAI Whisper function to convert your audio recording to a summary")
st.text('revsied by Dr. Tseng  @zinojeng')


# API Key Management: Fetch API key from environment variable
api_key = st.text_input(
      label="OpenAI API Key", 
      placeholder="Ex: sk-2twmA8tfCb8un4...", 
      key="openai_api_key", 
      help="You can get your API key from https://platform.openai.com/account/api-keys/")
os.environ["OPENAI_API_KEY"] = api_key


# Function to split long text into chunks
def split_text(text, max_length):
    """Split text into smaller chunks for processing."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Function to process a single chunk of text
def process_text(text, type):
    """Process a single chunk of text using GPT-3."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": type + text
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Function to process long text
def process_long_text(long_text, type):
    """Process long text by splitting it into smaller chunks."""
    text_list = split_text(long_text, 1200)
    processed_text_list = [process_text(text, type) for text in text_list]
    return "".join(processed_text_list)

# Streamlit UI
st.title('Audio to Transcript and Summary')

uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    # Resource Management: Use tempfile for storing the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(uploaded_file.read())
        audio_file_path = f.name
    
    try:
        audio_file = AudioSegment.from_file(audio_file_path)
        chunk_size = 100 * 1000  # 100 seconds
        chunks = [audio_file[i:i + chunk_size] for i in range(0, len(audio_file), chunk_size)]

        transcript = ""
        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                chunk.export(f.name, format="wav")
                temp_audio_file_path = f.name
            
            try:
                result = openai.Audio.transcribe("whisper-1", temp_audio_file_path)
                transcript += result["text"]
            except Exception as e:
                st.error(f"An error occurred while transcribing: {e}")
            finally:
                os.remove(temp_audio_file_path)

        processed_transcript = process_long_text(transcript, "Processing Transcript")
        processed_summary = process_long_text(processed_transcript, "Generating Summary")

        st.markdown("## Transcript")
        st.write(processed_transcript)

        st.markdown("## Summary")
        st.write(processed_summary)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        os.remove(audio_file_path)  # Remove the temp audio file
