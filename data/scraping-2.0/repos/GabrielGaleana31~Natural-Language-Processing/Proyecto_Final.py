import streamlit as st
import openai
import whisper

#Definimos la API key de OpenAI y el modelo de Whisper (GPT-3)
openai.api_key = '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'
model = whisper.load_model("base")

#Funciones para transcribir el audio usando el modelo 
def transcribe_audio(model, file_path):
    transcript = model.transcribe(file_path)
    return transcript['text']

#Funcion para resumir el texto a partir un base prompt y el input del usuario
def CustomChatGPT(user_input):
    messages = [{"role": "system", "content": "You are an office administer, summarize the text in key points"}]
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    return ChatGPT_reply


#Titulo de la app
st.markdown("<h1 style='text-align: center; color: grey;'>Transcription and summarization tool</h1>", unsafe_allow_html=True)

#Descripcion de la app
st.markdown("<h5 style='text-align: justify; color: white;'>Hi! I can assist you in transcribing or summarizing any speech you'd like. Just upload the file (only the m4a extension is supported) and click  on the transcribe button or on the summarize button. If you wish to transcribe or summarize another speech, simply delete the current file.</h5>", unsafe_allow_html=True)

#Advertencia
st.markdown("<h6 style='text-align: justify; color: red;'>Warning: Please be cautious regarding the output text from the model, verify the information, and refrain from sharing confidential files.</h6>", unsafe_allow_html=True)

#Definimos el boton para subir el archivo
uploaded_file = st.file_uploader("Choose a file")

#Definimos el boton para resumir el archivo
button = st.button("Transcribe", type="primary")

button_2 = st.button("Summarize", type="primary")

#Si el usuario sube un archivo y presiona el boton, se ejecuta el codigo
if uploaded_file is not None and button:

    #Guardamos el archivo en el directorio
    with open("uploaded_file_name.m4a", "wb") as f:
        f.write(uploaded_file.read())
    
    #Definimos el path del archivo
    path =  'uploaded_file_name.m4a'

    
    #Transcribimos el audio y resumimos el texto
    transcription = transcribe_audio(model, path)
    summary = CustomChatGPT(transcription)

    #Mostramos el resumen
    with st.chat_message("assistant"):
        st.write('The transcripted text is: \n', '\n', transcription)

        
    
if uploaded_file is not None and button_2:
    #Guardamos el archivo en el directorio
    with open("uploaded_file_name.m4a", "wb") as f:
        f.write(uploaded_file.read())
    
    #Definimos el path del archivo
    path =  'uploaded_file_name.m4a'

    
    #Transcribimos el audio y resumimos el texto
    transcription = transcribe_audio(model, path)
    summary = CustomChatGPT(transcription)

    #Mostramos el resumen
    with st.chat_message("assistant"):
        st.write('The summary is: \n', '\n', summary)    


