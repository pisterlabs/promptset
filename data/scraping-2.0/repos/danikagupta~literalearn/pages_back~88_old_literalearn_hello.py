import streamlit as st
aaa=""" 
from audio_recorder_streamlit import audio_recorder
import os
from openai import OpenAI
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import pandas as pd
from google.cloud import speech
from google.oauth2 import service_account
import base64
import json


def transcribe_audio_whisper(file_path,language_iso):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file, 
        language=language_iso,
        response_format="text"
    )
    print(f"Transcript: {transcript}")
    return transcript

def transcribe_audio(file_path,language_iso):
    gcs_credentials = st.secrets["connections"]["gcs"]
    credentials = service_account.Credentials.from_service_account_info(gcs_credentials)
    client = speech.SpeechClient(credentials=credentials)
    with open(file_path, 'rb') as audio_file:
        content = audio_file.read()
    # Configure the request with the language and audio file
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        #encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language_iso,
        audio_channel_count=2,
    )
    # Send the request to Google's Speech-to-Text API
    google_response = client.recognize(config=config, audio=audio)
    response=""
    for result in google_response.results:
        response+=result.alternatives[0].transcript

    print(f"Response: {google_response}")
    print(f"Response.results: {google_response.results}")
    st.sidebar.markdown(f"Response: {google_response}")
    # Print the transcription of the first alternative of the first result
    for result in google_response.results:
        print(f"Result: {result}")
        print(f"Result.alternatives: {result.alternatives}")
        print(f"Result.alternatives[0]: {result.alternatives[0]}")
        print("Transcription: {}".format(result.alternatives[0].transcript))
    return response


def bleu(hypothesis, reference):
  # Tokenize the hypothesis and reference strings.
  hypothesis_tokens = word_tokenize(hypothesis)
  reference_tokens = word_tokenize(reference)
  print(f"Reference tokens: {reference_tokens}, hypothesis tokens: {hypothesis_tokens}, reference: {reference}, hypothesis: {hypothesis}")
  st.markdown(f"Reference tokens: {reference_tokens}, hypothesis tokens: {hypothesis_tokens}, reference: {reference}, hypothesis: {hypothesis}")
  # Calculate the BLEU score.
  smoothie=SmoothingFunction().method1
  bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens,smoothing_function=smoothie)
  return bleu_score

def function_print_similarity_score(str1: str, str2: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    functions = [{
        "name": "print_similarity_score",
        "description": "A function that prints the similarity score of two strings",
        "parameters": {
            "type": "object",
            "properties": {
                "similarity_score": {
                    "type": "integer",
                    "enum": [1,2,3,4,5,6,7,8,9,10],
                    "description": "The similarity score."
                },
            }, "required": ["similarity_score"],
        }
    }]
    """
    llm_input=f"""
    You are a language reviewer responsible for reviewing the similarity of two sentences.
    Please note that the specifc words and word-order are important, not just the meaning.
    On a scale of 1-10, with 10 being the most similar, how similar are these: "{str1}", and "{str2}".
    """

    aaa="""
    messages = [{"role": "user", "content": llm_input}]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, functions=functions, function_call={"name": "print_similarity_score"})
    #print(f"Response is: {response}")
    function_call = response. choices [0].message.function_call
    #print(f"Function call is: {function_call}")
    argument = json.loads(function_call.arguments)
    #print(f"Response function parameters are: {argument}")
    print(f"For inputs {str1} and {str2}, the similarity score is: {argument}")
    return argument

# Removing for the time-being as we change the input file data @ st.cache_data
def getCSV():
    return pd.read_csv("assets/sentences.csv")

def generate_download_link(audio_file_path):
    with open(audio_file_path, "rb") as file:
        base64_file = base64.b64encode(file.read()).decode()
    href = f'<a href="data:file/wav;base64,{base64_file}" download="your_audio.wav"><img src="https://img.icons8.com/emoji/48/000000/play-button-emoji.png"/></a>'
    return href


def getSentence(language,difficulty):
    df=getCSV()
    with st.sidebar.expander("Show more"):
      st.dataframe(df)
      df=df[df["language"]==language]
      df=df[df["difficulty"]==difficulty]
      rec=df.sample().iloc[0]
      sentence=rec["sentence"]
      audiofile=rec["audiofile"]
      st.markdown(f"## Sentence={sentence}")
      fname=os.path.join(os.getcwd(),"audiofiles",audiofile)
      af = open(fname, 'rb')
      audiobytes = af.read()
    return sentence,audiobytes
    
#
# Main code
#
languages={"हिंदी":"hi","English":"en","മലയാളം":"ml","සිංහල":"si"}
main_instruction={"hi":"साफ़ से बोलें","en":"Speak clearly","ml":"വ്യക്തമായി സംസാരിക്കുക","si":"පැහැදිලිව කතා කරන්න"}

nltk_data_path = os.path.join(os.getcwd(),"nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

st.sidebar.title("LiteraLearn")
st.sidebar.image("assets/icon128px-red.png")
language_select=st.sidebar.selectbox("Language",options=languages.keys())
language_iso=languages[language_select]
instruction=main_instruction[language_iso]
sentence,audiobytes=getSentence(language_iso,1)
st.markdown(f"## {instruction}")
col1,col2=st.columns(2)
col1.markdown(f"{sentence}")
col2.audio(audiobytes, format="audio/wav")

path_myrecording = os.path.join(os.getcwd(),"audiofiles","myrecording.wav")
audio_bytes = audio_recorder(text="")
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with open(path_myrecording, mode='bw') as f:
        f.write(audio_bytes)
        f.close()
    transcription = transcribe_audio(path_myrecording,language_iso)
    st.markdown(f"Transcription: {transcription}")
    sc=bleu(transcription,sentence)
    sc2=function_print_similarity_score(transcription,sentence)
    st.markdown(f"BLEU score: {sc}")
    st.markdown(f"OpenAI score: {sc2}")
 """