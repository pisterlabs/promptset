import streamlit as st
import speech_recognition as sr
import openai
from medisearch_client import MediSearchClient
import uuid
from PIL import Image
import subprocess
import os
import PyPDF2
from io import BytesIO
import warnings

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("bg.png")

# Custom CSS for setting the background image
page_bg_img = f"""
<style>
body {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
}}

.stApp {{
    background: rgba(0, 0, 0, 0);  /* Adjust this as needed for the header */
}}
</style>
"""

# Display the background image using custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add margin-top style to move the header up by 40 pixels
header_style = """
<h1 style='text-align: center; color: red; margin-top: -80px;'>ABU - THE BOT</h1>
"""
st.markdown(header_style, unsafe_allow_html=True)

header_style = """
<h2 style='text-align: center; color: black; margin-top: -40px;'>Your Healthcare Bot</h2>"""
st.markdown(header_style, unsafe_allow_html=True)

# Display a GIF from a file in your working directory
st.image('abu.gif', caption='Your Virtual Assistant', use_column_width=True)

# OR

# Display a GIF from a URL

def hide_fullscreen_button():
    css = '''
    <style>
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)

# Call the function before displaying the image
hide_fullscreen_button()

# Custom CSS to style buttons (black color, larger size, and bold text)
button_style = """
<style>
div[data-testid="stButton"] button {
    background-color: black;
    color: white;
    padding: 16px 24px; /* Adjust padding for larger buttons */
    font-size: 18px;   /* Adjust font size for larger buttons */
    font-weight: bold; /* Make text bold */
}
</style>
"""

# Apply custom button styling
st.markdown(button_style, unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown('<style>div[data-testid="stHorizontalBlock"]{text-align: center;}</style>', unsafe_allow_html=True)
col2.markdown('<style>div[data-testid="stHorizontalBlock"]{text-align: center;}</style>', unsafe_allow_html=True)
col3.markdown('<style>div[data-testid="stHorizontalBlock"]{text-align: center;}</style>', unsafe_allow_html=True)
col4.markdown('<style>div[data-testid="stHorizontalBlock"]{text-align: center;}</style>', unsafe_allow_html=True)
col5.markdown('<style>div[data-testid="stHorizontalBlock"]{text-align: center;}</style>', unsafe_allow_html=True)


import pyttsx3

# Set your OpenAI API key
api_key = "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"
openai.api_key = api_key

def generate_gpt3_response(text):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        return str(e)

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Recording... Speak something.")
        audio_data = r.listen(source)
        return audio_data

def main():
    
    
    if col2.button("General"):
        st.write("Recording... Click 'Stop Recording' when finished.")
        audio_data = record_audio()
        st.write("Recording complete!")

        r = sr.Recognizer()
        try:
            text = r.recognize_google(audio_data)
            st.subheader("Transcription:")
            st.write(text)

            response = generate_gpt3_response(text)
            st.write(response)
            engine = pyttsx3.init()
            st.image('speak.gif', caption='ABU - THE BOT', use_column_width=True)
            engine.say(response)
            engine.runAndWait()
            st.image('abu.gif', caption='ABU - THE BOT', use_column_width=True)

        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")

    if col3.button("SIGN"):
    
        desired_directory = r'E:\projects\chiro\action\env'
        os.chdir(desired_directory)    

        script_path = 'main.py' 

        try:            
            os.system(f'python {script_path}')

            with open('E:\\projects\\chiro\\env\\output.txt', 'r') as file:
                code = file.read()
                print(code)

            text = code
            if "Father" in text:
                text = "i had a head pain need help"

            if "Sorry" in text:
                text = "I am having stomach pain"    
            st.subheader("Transcription:")
            st.write(text)

            response = generate_gpt3_response(text)
            st.write(response)
            engine = pyttsx3.init()
            st.image('speak.gif', caption='ABU - THE BOT', use_column_width=True)
            engine.say(response)
            engine.runAndWait()
            st.image('abu.gif', caption='ABU - THE BOT', use_column_width=True)

        except Exception as e:
            st.error(f"Error running SIGN: {str(e)}")
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    
                
    if col4.button("Report"):
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        def extract_text_from_pdf(uploaded_file):
            if uploaded_file is not None:
                warnings.filterwarnings("ignore", category=PyPDF2.utils.PdfReadWarning)
                pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
                page_content = []
                print(pdf_reader.numPages)

                for page_num in range(pdf_reader.numPages):
                    print(pdf_reader.numPages)
                    page = pdf_reader.getPage(page_num)
                    text = page.extractText()
                    page_content.append(text)
                    print(text)

                return page_content

        if uploaded_file:
            st.write("File uploaded successfully!")
            extracted_text = extract_text_from_pdf(uploaded_file)
            for page_num, text in enumerate(extracted_text, start=1):
                    st.write(f"Page {page_num}:\n{text}\n")
                    api_key = "8590291a-8835-463b-a9e8-c8e1e08535ec"
                    conversation_id = str(uuid.uuid4())
                    client = MediSearchClient(api_key=api_key)
                    responses = client.send_user_message(conversation=[text], 
                                            conversation_id=conversation_id,
                                            should_stream_response=True,
                                            language="English")
                    st.write(responses)
                    for response in responses:
                        if response["event"] == "llm_response":
                            llm_answer = response["text"]
                            print(llm_answer)
                    st.write(llm_answer)
                    print(llm_answer)
                    engine = pyttsx3.init()
                    engine.say(llm_answer)
                    st.image('speak.gif', caption='ABU - THE BOT', use_column_width=True)
                    engine.runAndWait()
                    break

            
if __name__ == "__main__":
    main()
