import os
import sys
import datetime
import openai
import streamlit as st

from audio_recorder_streamlit import audio_recorder

working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)

def transcribe(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript

def summarize(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=(
            f"Please do what you are asked to do with the following text:\n"
            f"{text}"
        ),
        temperature=0.5,
        max_tokens=560,
    )

    return response.choices[0].text.strip()

# Add a text input widget for the user to enter their API key in the Streamlit app's sidebar
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Set up the OpenAI API using the user-provided API key
if openai_api_key:
    openai.api_key = openai_api_key

st.image("https://asesorialinguistica.online/wp-content/uploads/2023/04/Secretary-GPT.png")

st.text("Click on the microphone and tell your GPT secretary what to type.")

st.sidebar.title("Secretary GPT")

# Explanation of the app
st.sidebar.markdown("""
## Instructions
0. Make sure your browser allows microphone access to this site.
1. Choose to record audio or upload an audio file.
2. To begin, tell your secretary what to record: an email, a report, an article, an essay, etc.
3. If you record audio, click on the microphone icon to start and to finish.
4. If you are uploading an audio file, select the file from your device and upload it.
5. Click on Transcribe. The waiting time is proportional to the recording time.
6. The transcription appears first and then the request.
7. Download the generated document in text format.
-  By Moris Polanco
        """)

# tab record audio and upload audio
tab1, tab2 = st.columns(2)

with tab1:
    audio_bytes = audio_recorder(pause_threshold=300)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # save audio file to mp3
        with open(f"audio_{timestamp}.mp3", "wb") as f:
            f.write(audio_bytes)

with tab2:
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])

    if audio_file:
        # st.audio(audio_file.read(), format={audio_file.type})
        timestamp = timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # save audio file with correct extension
        with open(f"audio_{timestamp}.{audio_file.type.split('/')[1]}", "wb") as f:
            f.write(audio_file.read())

if st.button("Transcribe"):
    # check if audio file exists
    if not any(f.startswith("audio") for f in os.listdir(".")):
        st.warning("Please record or upload an audio file first.")
    else:
        # find newest audio file
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
        )
        
        # transcribe
        audio_file = open(audio_file_path, "rb")

        transcript = transcribe(audio_file)
        text = transcript["text"]

        st.subheader("Transcript")
        st.write(text)

        # summarize
        summary = summarize(text)

        st.subheader("Document")
        st.write(summary)

        # save transcript and summary to text files
        with open("transcript.txt", "w") as f:
            f.write(text)

        with open("summary.txt", "w") as f:
            f.write(summary)

        # download transcript and summary
        st.download_button('Download Document', summary)

# delete audio and text files when leaving app
if not st.session_state.get('cleaned_up'):
    files = [f for f in os.listdir(".") if f.startswith("audio") or f.endswith(".txt")]
    for file in files:
        os.remove(file)
    st.session_state['cleaned_up'] = True
