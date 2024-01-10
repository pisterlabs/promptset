''' This is the subpage for the AssemblyAI agent.
'''

#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------#
import os
import streamlit as st
import tempfile
from main import rec_streamlit, speak_answer, get_transcript_whisper
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit import cache_resource
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from credentials import OPENAI_API_KEY, ASSEMBLYAI_API_KEY
import pytube
import requests
import time


#%% ----------------------------------------  LANGCHAIN & ASSEMBLYAI PRELOADS -----------------------------------------------------#
# --- LangChain ---
embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=OPENAI_API_KEY)
llm_csv = OpenAI(openai_api_key=OPENAI_API_KEY)

# --- AssemblyAI ---
base_url = "https://api.assemblyai.com/v2"
headers = {
    "authorization": ASSEMBLYAI_API_KEY
}


# --------------------  YOUTUBE VIDEO DOWNLOADER  -------------------- #
def download_video(url):
    try:
        youtube = pytube.YouTube(url)
        video = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if video:
            video_title = video.title
            video.download(filename=video_title)
            print("Video downloaded successfully.")
            
            # Get the video path
            video_path = os.path.join(os.getcwd(), video_title)

            return video_path,video_title
        else:
            print("No suitable video format found.")
    except pytube.exceptions.PytubeError as e:
        print(f"Error: {str(e)}")


# --------------------  SETTINGS  -------------------- #
st.set_page_config(page_title="Home", layout="wide")
st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""", unsafe_allow_html=True)

# --------------------- HOME PAGE -------------------- #
st.title("LANGCHAIN ASSEMBLYAI-AGENT 🤖")
st.write("""Use the power of LLMs and Assembly AI to interact with Video and audio files. By uploading a video, audio file or a Youtube link, 
        you can interact with the content of the file.The video will be transcribed and the text will be used to interact with the LLMs. 
        Speakers will also be detected""")

st.write("Let's start interacting with AssemblyAI!")

# ----------------- SIDE BAR SETTINGS ---------------- #
st.sidebar.subheader("Settings:")
tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
ner_enabled = st.sidebar.checkbox("Enable NER in Response", value=False)

# ------------------ FILE UPLOADER ------------------- #
st.sidebar.subheader("File Uploader:")

uploaded_files = st.sidebar.file_uploader("Choose files", type=["mp3", "mp4"],
                                          accept_multiple_files=True)
st.sidebar.metric("Number of files uploaded", len(uploaded_files))
st.sidebar.color_picker("Pick a color for the answer space", "#C14531")

# ------------------ YOUTUBE VIDEO HANDLER ------------------- #
st.subheader("Youtube Link:")
youtube_link = st.text_input("Enter a Youtube link")

@cache_resource
def youtube_upload(youtube_link):
  if youtube_link:
      youtubelink, videoname = download_video(youtube_link)
      st.write(f"Video downloaded successfully: {videoname}")
      st.video(youtubelink)
      mesUnimake = st.success(f'"{videoname}" uploaded successfully! Please wait a moment while we upload the video to our servers...')
            
      # upload the video
      with open(youtubelink, "rb") as f:
          response = requests.post(base_url + "/upload",
                                  headers=headers,
                                  data=f)
          upload_url = response.json()["upload_url"]
      data = {
      "audio_url": upload_url,
      "auto_chapters": True,
      "speaker_labels": True,
      "sentiment_analysis": True
      }
      mesUnimake.empty()
      mesUnimake = st.success("Upload complete!")
      time.sleep(1)
      mesUnimake.empty()
      mesUnimake = st.success("Transcribing video...")

      url = base_url + "/transcript"
      response = requests.post(url, json=data, headers=headers)

      transcript_id = response.json()['id']
      polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

      while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result['status'] == 'completed':
          transcript_text = transcription_result['text']
          utter = transcription_result['utterances']

          # Iterate through each utterance and print the speaker and the text they spoke
          for utterance in utter:
              speaker = utterance['speaker']
              text = utterance['text']
              print(f"Speaker {speaker}: {text}")

              # append to a txt file
              with open(f'{videoname}_speaker_labels.txt', 'a') as f:
                  f.write(f"Speaker {speaker}: {text}\n")
          
          mesUnimake.empty()
          mesUnimake = st.success("Transcription complete!")
          time.sleep(3)
          mesUnimake.empty()
          
          return videoname, utter

        elif transcription_result['status'] == 'error':
          mesUnimake.empty()
          mesUnimake = st.error("Transcription failed!")
          raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

        else:
          time.sleep(3)
          



if youtube_link:
  videoname, utterances = youtube_upload(youtube_link)
  st.write("Video ready to be communicated with!")
  #Save the text with speaker labels to a txt file
  with open(f'{videoname}_speaker_labels.txt', 'w') as f:
      for utterance in utterances:
          speaker = utterance['speaker']
          text = utterance['text']
          f.write(f"Speaker {speaker}: {text}\n")

  #import and store the txt file
  with open(f'{videoname}_speaker_labels.txt', 'r') as f:
      text = f.read()

  if text:
    try:
      # --- Display the file content as code---
      with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=False):
          st.subheader("Uploaded Document:")
          st.write(text)

    except Exception as e:
      st.write("Error reading file:", e)


  loader = TextLoader(f'{videoname}_speaker_labels.txt')

  # ------------------- LANGCHAIN ------------------- #
  documents = loader.load()

  #Get your splitter ready
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
  texts = text_splitter.split_documents(documents)
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
  docsearch = Chroma.from_documents(texts, embeddings)
  qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", vectorstore=docsearch)

  # --------------------- USER INPUT --------------------- #
  user_input = st.text_area("")

  # If record button is pressed, rec_streamlit records and the output is saved
  audio_bytes = rec_streamlit()

  # ------------------- TRANSCRIPTION -------------------- #
  if audio_bytes or user_input:

      if audio_bytes:
          try:
              with open("langchain_assemblyai_audio.wav", "wb") as file:
                  file.write(audio_bytes)
          except Exception as e:
              st.write("Error recording audio:", e)
          transcript = get_transcript_whisper("langchain_assemblyai_audio.wav")
      else:
          transcript = user_input

      st.write("**Recognized:**")
      st.write(transcript)

      if any(word in transcript for word in ["abort recording"]):
          st.write("... Script stopped by user")
          exit()
      # ----------------------- ANSWER ----------------------- #
      with st.spinner("Fetching answer ..."):
          time.sleep(6)

      # Use the CSV agent to answer the question
      query = transcript
      answer  = qa.run(query)
      st.write("AI Response:", answer)
      speak_answer(answer, tts_enabled)
      st.success("**Interaction finished**")


# ------------------- FILE HANDLER ------------------- #
if uploaded_files:
  file_index = st.sidebar.selectbox("Select a file to display", options=[f.name for f in uploaded_files])
  selected_file = uploaded_files[[f.name for f in uploaded_files].index(file_index)]
  file_extension = selected_file.name.split(".")[-1]

  # Save the file to a temporary directory
  with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(selected_file.read())
    tmp_filename = f.name

  # load the file
  if file_extension == "mp3":
    st.audio(tmp_filename)
  elif file_extension == "mp4":
    st.video(tmp_filename)

  with open(tmp_filename, "rb") as f:
    response = requests.post(base_url + "/upload",
                            headers=headers,
                            data=f)

    upload_url = response.json()["upload_url"]


  data = {
  "audio_url": upload_url,
  "auto_chapters": True,
  "speaker_labels": True,
  "sentiment_analysis": True
  }

  url = base_url + "/transcript"
  response = requests.post(url, json=data, headers=headers)

  transcript_id = response.json()['id']
  polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

  while True:
    transcription_result = requests.get(polling_endpoint, headers=headers).json()

    if transcription_result['status'] == 'completed':
      transcript_text = transcription_result['text']
      utterances = transcription_result['utterances']

      # Iterate through each utterance and print the speaker and the text they spoke
      for utterance in utterances:
          speaker = utterance['speaker']
          text = utterance['text']
          print(f"Speaker {speaker}: {text}")

          # append to a txt file
          with open('morning_session.txt', 'a') as f:
              f.write(f"Speaker {speaker}: {text}\n")

      break

    elif transcription_result['status'] == 'error':
      raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

    else:
      time.sleep(3)
  
  #Save the text with speaker labels to a txt file
  with open(f'{selected_file.name}_speaker_labels.txt', 'w') as f:
      for utterance in utterances:
          speaker = utterance['speaker']
          text = utterance['text']
          f.write(f"Speaker {speaker}: {text}\n")

  #import and store the txt file
  with open(f'{selected_file.name}_speaker_labels.txt', 'r') as f:
      text = f.read()
  
  if text:
    try:
      # --- Display the file content as code---
      with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=False):
          st.subheader("Uploaded Document:")
          st.write(text)

    except Exception as e:
      st.write("Error reading file:", e)
  
  from langchain.document_loaders import TextLoader
  loader = TextLoader(f'{selected_file.name}_speaker_labels.txt')
  # ------------------- LANGCHAIN ------------------- #
  documents = loader.load()

  #Get your splitter ready
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
  texts = text_splitter.split_documents(documents)
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
  docsearch = Chroma.from_documents(texts, embeddings)
  qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", vectorstore=docsearch)

  # --------------------- USER INPUT --------------------- #
  user_input = st.text_area("")

  # If record button is pressed, rec_streamlit records and the output is saved
  audio_bytes = rec_streamlit()

  # ------------------- TRANSCRIPTION -------------------- #
  if audio_bytes or user_input:

      if audio_bytes:
          try:
              with open("langchain_assemblyai_audio.wav", "wb") as file:
                  file.write(audio_bytes)
          except Exception as e:
              st.write("Error recording audio:", e)
          transcript = get_transcript_whisper("langchain_assemblyai_audio.wav")
      else:
          transcript = user_input

      st.write("**Recognized:**")
      st.write(transcript)

      if any(word in transcript for word in ["abort recording"]):
          st.write("... Script stopped by user")
          exit()
      # ----------------------- ANSWER ----------------------- #
      with st.spinner("Fetching answer ..."):
          time.sleep(6)

      # Use the CSV agent to answer the question
      query = transcript
      answer  = qa.run(query)
      st.write("AI Response:", answer)
      speak_answer(answer, tts_enabled)
      st.success("**Interaction finished**")