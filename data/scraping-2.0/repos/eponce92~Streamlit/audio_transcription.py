import streamlit as st
import openai

from pytube import YouTube  # You can also use youtube_dl
import re
from streamlit_extras.stylable_container import stylable_container  # Import stylable_container
from typing import Optional

def validate_user(username: str, password: str) -> Optional[str]:
    for user in st.secrets["users"]:
        if user["username"] == username and user["password"] == password:
            return username
    return None

# Function to download YouTube video audio
def download_audio(youtube_url):
    yt = YouTube(youtube_url)
    sanitized_title = re.sub(r'[^\w\s]', '', yt.title)
    sanitized_title = re.sub(r'\s+', '_', sanitized_title)
    audio_file_path = f"{sanitized_title}.webm"
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(filename=audio_file_path)
    return audio_file_path

# Function to transcribe audio using Whisper API
def whisper_transcribe(audio_file_path, api_key):
    client = openai.OpenAI(api_key=api_key)
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
    return transcript
    
# Function to continue the chat conversation

def continue_chat(api_key, messages, model):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content



def main():
    # Set the page config at the beginning of your script
    st.set_page_config(
            page_title="Audio Transcriber GPT",
            page_icon="https://raw.githubusercontent.com/eponce92/Streamlit/main/audio-transcript-icon.png",  # Direct link to your favicon image
            layout="centered"
        )
    

    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    
    # Add this line to initialize authenticated_user
    if 'authenticated_user' not in st.session_state:
        st.session_state.authenticated_user = None

    # Use a different image as part of the title
        st.markdown(
            f'''
            <div style="display: flex; align-items: center;">
                <h1 class="streamlit-header" style="margin:0;">Chat with your video </h1>
                <img src="https://raw.githubusercontent.com/eponce92/Streamlit/main/audio-transctipt-ilustration.png" style="height:auto;width:260px;margin-left: 10px;">
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    
        # Description
        st.markdown(
            """
            ## What does this app do? 🤔
            - This app lets you transcribe audio from a YouTube video or an uploaded audio file.
            - Once transcribed, you can interact with the GPT model to ask questions, summarize the content, or discuss it in a chat format.
            
            ## How to Use 🛠️
            1. **Log in**: Use the sidebar to login. You'll need a valid username and password.
            2. **Select a GPT model**: Choose the GPT model you'd like to use from the dropdown.
            3. **Provide Audio**: 
                - **Option A**: Paste a YouTube video URL.
                - **Option B**: Upload an audio file.
            4. **Transcribe**: Click the 'Transcribe' button. 
            5. **Chat**: Use the chat box at the bottom to ask questions or discuss the content with the GPT model.
            
            ---
            """
        )


    with st.sidebar.form("login_form"):
        username_input = st.text_input("Username:")
        password_input = st.text_input("Password:", type="password")
        login_button = st.form_submit_button("Login")
    
        if login_button:
            authenticated_user = validate_user(username_input, password_input)
            if authenticated_user is not None:
                st.sidebar.success(f"Logged in as {authenticated_user}")
                st.session_state.authenticated_user = authenticated_user
            else:
                st.sidebar.error("Invalid username or password")






    if st.session_state.authenticated_user:
           # Use a different image as part of the title
        st.markdown(
            f'''
            <div style="display: flex; align-items: center;">
                <h1 class="streamlit-header" style="margin:0;">Audio Transcriber GPT</h1>
                <img src="https://raw.githubusercontent.com/eponce92/Streamlit/main/audio-transctipt-ilustration.png" style="height:auto;width:260px;margin-left: 10px;">
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    
        # Description
        st.markdown(
            """
            ## What does this app do? 🤔
            - This app lets you transcribe audio from a YouTube video or an uploaded audio file.
            - Once transcribed, you can interact with the GPT model to ask questions, summarize the content, or discuss it in a chat format.
            
            ## How to Use 🛠️
            1. **Log in**: Use the sidebar to login. You'll need a valid username and password.
            2. **Select a GPT model**: Choose the GPT model you'd like to use from the dropdown.
            3. **Provide Audio**: 
                - **Option A**: Paste a YouTube video URL.
                - **Option B**: Upload an audio file.
            4. **Transcribe**: Click the 'Transcribe' button. 
            5. **Chat**: Use the chat box at the bottom to ask questions or discuss the content with the GPT model.
            
            ---
            """
        )
            
    
        # Initialize session state
        if 'youtube_video_embed_url' not in st.session_state:
            st.session_state['youtube_video_embed_url'] = None
    
       
        # Dropdown for GPT model selection
        gpt_model = st.selectbox(
            "Select GPT model:",
            ("gpt-4-1106-preview", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-1106")
        )

        openai_api_key = st.secrets["api_key"]
    
        # YouTube URL or Upload File
        youtube_url = st.text_input("Enter YouTube Video URL:")
        uploaded_file = st.file_uploader("Or upload an audio file:", type=["mp3", "wav", "webm"])
    
        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{"role": "system", "content": "Use this audio transcription as context to chat with the user. The user might ask you to summarize or questions about the content of the transcription and you should answer based on this information."}]
    


        if st.button('Transcribe'):
            st.session_state.transcribe_button = True
            
           
        
        if st.session_state.get('authenticated_user') and st.session_state.get('transcribe_button', False):


        
            with st.spinner("Transcribing..."):

                try:
                    if not openai_api_key or (not youtube_url and not uploaded_file):
                        st.warning("Please fill in all required fields.")
                    else:
                        try:
                            st.info("Preparing to download and transcribe audio...")  # Added info
                            if youtube_url:
                                # Download the video and extract audio for transcription
                                audio_file_path = download_audio(youtube_url)
                                st.session_state['youtube_video_embed_url'] = f"https://www.youtube.com/embed/{YouTube(youtube_url).video_id}"
                            
                            else:
                                # Use the uploaded audio file for transcription
                                audio_file_path = uploaded_file.name
                                with open(audio_file_path, "wb") as f:
                                    f.write(uploaded_file.read())
                        
                            st.info("Performing transcription...")  # Added info
                            # Proceed with transcription
                            transcription = whisper_transcribe(audio_file_path, openai_api_key)
                            
                            # Show transcription
                            with st.expander("Show Transcription"):
                                with stylable_container(
                                    "codeblock",
                                    """
                                    code {
                                        white-space: pre-wrap !important;
                                    }
                                    """,
                                ):
                                    st.code(transcription)
                        


                            # Add transcription to message history only if it's not already present
                            if {"role": "assistant", "content": f"Transcription: {transcription}"} not in st.session_state['messages']:
                                st.session_state['messages'].append({"role": "assistant", "content": f"Transcription: {transcription}"})

                            
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        
    
        # Chat Interface
        if 'messages' in st.session_state and len(st.session_state['messages']) > 1:
        
            st.write("## Continue Chatting with GPT")
    
            # Display previous messages
            for message in st.session_state['messages']:
                role = message["role"]
                content = message["content"]
                with st.chat_message(role):
                    st.write(content)
                    
            # Initialize session state for chat if it doesn't exist
            if 'messages' not in st.session_state:
                st.session_state['messages'] = []
            
            # Initialize last_user_input if it doesn't exist
            if 'last_user_input' not in st.session_state:
                st.session_state.last_user_input = None
            
            # User input
            user_input = st.chat_input("Type your message", key="unique_key_for_chat_input")
            
            if user_input and user_input != st.session_state.last_user_input:
                # Add user message to message history
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state.last_user_input = user_input  # Update last_user_input
            
                # Get GPT response
                gpt_response = continue_chat(openai_api_key, st.session_state['messages'], gpt_model)
            
                # Add GPT message to message history
                st.session_state['messages'].append({"role": "assistant", "content": gpt_response})
            
                with st.chat_message("assistant"):
                    st.write(gpt_response)
            
                # Force a rerun to update the chat interface
                st.rerun()
                    
              
    
            # Display video if available in session state
            if st.session_state['youtube_video_embed_url']:
                st.video(st.session_state['youtube_video_embed_url'])

if __name__ == "__main__":
    main()
