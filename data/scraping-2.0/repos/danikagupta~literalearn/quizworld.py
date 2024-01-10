import streamlit as st
from openai import OpenAI

from google.cloud import speech
from google.oauth2 import service_account

from audio_recorder_streamlit import audio_recorder

import os
import json

import cookiestore
import datastore


def transcribe_audio(file_path,language_iso,debugging):
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

    if debugging:
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
                    "description": "The similarity score."
                },
            }, "required": ["similarity_score"],
        }
    }]
    llm_input=f"""
    You are a language reviewer responsible for reviewing the similarity of two sentences.
    The user is being given a sentence, and asked to repeat the sentence themselves.
    As such, the scoring has to be lenient.
    Please note that the specifc words and word-order are important, not just the meaning.
    On a scale of 1-100, with 100 being the most similar, how similar are these: "{str1}", and "{str2}".
    """
    messages = [{"role": "user", "content": llm_input}]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, functions=functions, function_call={"name": "print_similarity_score"})
    #print(f"Response is: {response}")
    function_call = response. choices [0].message.function_call
    #print(f"Function call is: {function_call}")
    argument = json.loads(function_call.arguments)
    #print(f"Response function parameters are: {argument}")
    print(f"For inputs {str1} and {str2}, the similarity score is: {argument} and type is {type(argument)}")
    result=argument['similarity_score']
    return result



def ask_question(user_sub,user_name, selected_question,audiofile,language, level, languages, debugging):
    if debugging:
        st.markdown(f"## Ask Question: {selected_question} for user {user_sub}") 
    main_instruction={"hi":"साफ़ से बोलें","en":"Speak clearly","ml":"വ്യക്തമായി സംസാരിക്കുക","si":"පැහැදිලිව කතා කරන්න"}  
    language_iso=language
    instruction=datastore.get_i18n('speakClearly',language_iso,debugging)
    inverted_lang = {value: key for key, value in languages.items()}
    #main_instruction[language_iso]
    sentence=selected_question
    st.sidebar.write(f"{user_name} {inverted_lang[language_iso]} {level}")
    # st.divider()
    st.markdown(f"# {instruction}")    
    st.markdown(f"## {sentence}")
    # Single audio at this time; will investigate TTS. 
    fname=os.path.join(os.getcwd(),"audiofiles",audiofile)
    af = open(fname, 'rb')
    audiobytes = af.read()
    button_id = "green_button"

    button_css = f"""
    <style>
    #{button_id} {{
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
    }}
    </style>
    """

# Inject custom CSS with the HTML
    st.markdown(button_css, unsafe_allow_html=True)
    bu = st.button("🙋 Help",key=button_id)
    if bu:
        st.audio(audiobytes, format="audio/wav")
    # music_code=cookiestore.get_music_code(audiofile)
    # st.markdown(music_code, unsafe_allow_html=True)

    path_myrecording = os.path.join(os.getcwd(),"audiofiles","myrecording.wav")
    audio_bytes = audio_recorder(text="")
    if audio_bytes:
        with open(path_myrecording, mode='bw') as f:
            f.write(audio_bytes)
            f.close()
        transcription = transcribe_audio(path_myrecording,language_iso,debugging)
        sc2=function_print_similarity_score(transcription,sentence)
        st.sidebar.audio(audio_bytes, format="audio/wav")
        st.sidebar.markdown(f"Transcription: {transcription}")
        st.sidebar.markdown(f"Score: {sc2}") 
        return sc2
    else:
        return -1