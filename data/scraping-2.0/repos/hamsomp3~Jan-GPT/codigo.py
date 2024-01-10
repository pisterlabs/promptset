from openai import OpenAI
import streamlit as st
import os
import base64
from dotenv import load_dotenv
load_dotenv()
os.environ.get("OPENAI_API_KEY")

# Verifica si la clave API de OpenAI está disponible
if "OPENAI_API_KEY" in os.environ:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
else:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está definida")




def main():
    # Configuración de la página de Streamlit
    st.title("Welcome to JANGPT")

 
    # Sidebar for initial parameters
    with st.sidebar:
        st.title("Initial Parameters for Chat")
        st.session_state.temperature = st.slider("Select the temperature:", 0.0, 1.0, 0.1)
        st.session_state.model_box = st.selectbox("Select the model:", ("gpt-3.5-turbo", "gpt-4-1106-preview"))
        st.title("Initial Parameters for Audio")
        st.session_state.audio_model = st.selectbox("Select the model:", ("tts-1", "tts-1-hd"))
        st.session_state.audio_voice = st.selectbox("Select the voice:", ("alloy", "echo", "fable", "onyx", "nova", "shimmer"))
        st.session_state.start_chat = st.button("Start Chat")
    # Display a message to start the chat if the button is clicked
        
    if st.session_state.start_chat:
        st.success("Chat started! Use the input below to send a message.")

    # Crear una caja de texto para la entrada del usuario
    user_input = st.text_input("Escribe tu mensaje aquí:")
    
    def convert_audio(audiofile):
    # Open the audio file in binary mode
        with open(audiofile, 'rb') as file:
            audio_data = file.read()

        # Encode the binary data to base64
        b64 = base64.b64encode(audio_data).decode()

    # Embed the base64 string in the HTML audio tag
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md
    
    if st.button("Enviar"):
        # Enviar el mensaje del usuario a OpenAI y obtener la respuesta
        
        chat_completion = client.chat.completions.create(
        messages=[
            {
            "role": "user",
            "content": user_input,
            }
        ],
        model=st.session_state.model_box,
        )        
        

        audio_response = client.audio.speech.create(
            model=st.session_state.audio_model,
            voice=st.session_state.audio_voice,
            input=str(chat_completion.choices[0].message.content),
        )
        audio_file = "output.mp3"
        audio_response.stream_to_file(audio_file)

        #st.audio(audio_file)

        md = convert_audio(audio_file)
        st.markdown(md, unsafe_allow_html=True)

        st.text("Respuesta del chatbot:")
        st.write(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()