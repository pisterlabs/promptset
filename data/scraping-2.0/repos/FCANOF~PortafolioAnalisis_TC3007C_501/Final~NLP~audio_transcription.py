import streamlit as st
import openai
#https://medium.com/gitconnected/using-the-whisper-api-to-transcribe-audio-files-45fb36d1aa1b

import whisper
# brew install ffmpeg # you may need to install
import os

openai.api_key = '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'

global model 
model = whisper.load_model("base")

def transcribe_audio(model, file_path):
    transcript = model.transcribe(file_path)
    return transcript['text']

def CustomChatGPT(user_input):
    messages = [{"role": "system", "content": "You are an office administer, summarize the text in key points"}]
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    return ChatGPT_reply

# Streamlit app
def main():
    st.title("Transcripción de audio y resumen con GPT-3")
    st.markdown("Frida Cano Falcón - A01752953")
    st.markdown("Esta aplicación web transcribe un archivo de audio y genera un resumen del mismo utilizando GPT-3.")
    
    # Upload audio file
    audio_file = st.file_uploader("Inserta tu archivo de audio aquí", type=["mp3", "wav", "m4a"])
    
    if audio_file:
        # Guardar el archivo en la carpeta local
        file_path = os.path.join(os.getcwd(), audio_file.name)
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())
            st.success("Archivo guardado correctamente")
            
        # Reproductor del audio
        st.audio(audio_file, format="audio")
        
        # Transcripción del audio
        st.header("Transcripción:")
        transcription = transcribe_audio(model, audio_file.name)
        st.write(transcription)

        # Generar resumen con ChatGPT
        st.header("Resumen:")
        summary = CustomChatGPT(transcription)
        st.write(summary)

if __name__ == "__main__":
    main()