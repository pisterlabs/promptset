import streamlit as st
import speech_recognition as sr
import openai
from transformers import pipeline


api_key = '' 

# Function to convert audio to text
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioFile(audio_file)
    
    with audio_data as source:
        audio = recognizer.record(source)
    
    text = recognizer.recognize_google(audio, language="es-ES")
    return text

# Function to call the OpenAI endpoint v1/completions
def call_openai_api(prompt_input):
    openai.api_key = api_key
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= f"Resume este texto de manera completa pero breve: {prompt_input}",
        max_tokens=200
    )

    summary = response['choices'][0]['text'].strip() if 'choices' in response and len(response['choices']) > 0 else "No se pudo generar un resumen."
    return summary


# Streamlit config
st.header('Alexia Naredo A00830440')
st.title("Final Project: Natural Language Processing")


uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"])
st.write('Upon uploading your file, our system will generate a text transcription, an analysis of the original sentiment, and provide you with a a summary generated with AI')
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Transcribe and Process Text"):
        text = transcribe_audio(uploaded_file)
        st.header("Transcripción:")
        st.write(text)

        summary = call_openai_api(text)
        st.header("Summary (Generated with OpenAI):")
        st.write(summary)

        nlp = pipeline("sentiment-analysis")
        result = nlp(text)
        st.header("Sentiment Analysis")

        if result:
            label = result[0]["label"]
            score = result[0]["score"]

            sentiment_map = {
                "POSITIVE": "Positive",
                "NEGATIVE": "Negative",
                "NEUTRAL": "Neutral",
                # Puedes añadir más etiquetas si es necesario
            }

            if label in sentiment_map:
                description = sentiment_map[label]
            else:
                description = "Undefined"

            st.write(f"Sentiment: {description} (Score: {score})")
        else:
            st.write("Unable to process, please troubleshoot.")

    
        

