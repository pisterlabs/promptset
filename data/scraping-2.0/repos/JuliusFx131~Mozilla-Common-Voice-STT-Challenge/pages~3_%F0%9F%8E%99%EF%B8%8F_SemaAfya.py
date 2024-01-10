import io
import json
import time
import openai 
import requests
from utils import*
import numpy as np
import streamlit as st
from zipfile import ZipFile
from pydub import AudioSegment
#from audiorecorder import AudioRecorder
from audiorecorder import audiorecorder
#from audiorecorder import audiorecorder
openai.api_key=st.secrets["OPENAI_API_KEY"]

# configurations
st.set_page_config(
    page_title="SemaAfya",
    page_icon="🏥",
    layout="centered",
    #initial_sidebar_state="collapsed"
)

# Check if user is new or returning using session state.
# If user is new, show the toast message.
if 'first_time' not in st.session_state:
    message, icon = get_random_toast()
    st.toast(message, icon=icon)
    st.session_state.first_time = False

header_html = "<div style='text-align: left;'><img src='data:image/png;base64,{}' class='img-fluid' style='max-width: 700px;'></div>".format(
    img_to_bytes("Logo.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

# UI
#st.write("####")
#st.title("Karibu SemaAfya 🏥✨🤖") #✨🎙️
#st.title('SemaAfya🏥') 
#st.write('Huyu ni msaidizi wako wa kujifunza lugha ya kibinafsi. Hapa, unaweza kufanya mazoezi ya kuzungumza, kujadili makala, na kupata maoni ya papo hapo katika lugha uliyochagua! Kwa sasa, unaweza kufanya mazoezi ya Kijerumani, Kifaransa, Kiitaliano, Kihispania, Kireno, Kiholanzi, Kirusi, Kijapani, Kichina, Kihindi na Kiingereza katika muda halisi, bila mshirika.')
#st.markdown("---") 

# Sidebar content
#st.sidebar.header("**Pata zaidi kutoka🤖 SemaAfya**")
st.sidebar.title("🗝️ Maelekezo na Vidokezo")
st.sidebar.write("""
#### 📌 Jinsi ya kutumia:
1. Ikiwa ungetaka kutembelea kituo chetu cha afya, chagua **"Ndio"** kwenye swali "Je ungetaka kutuma suala lako la matibabu kwa daktari?Jaza nambari yako ya simu kwenye sanduku la maandishi lililoonekana. Ikiwa hauko tayari, chagua **"La"**.
2. **Rekodi** sauti yako kwa kutia alama kitufe cha maikrofoni. 
2. Unaweza pia weka sauti uliyo rekodi tiyari kwa kubofya kitufe cha **"Browse Files"**
3. Kusoma na kusikiliza sauti yako bofya kitufe **Tafsiri Sauti** halafu usubiri kidogo 
4. Utapata ushauri kutoka kwa msaidizi wako wa afya na unaweza pakua sauti na jibu la suala lako. 
""")
st.sidebar.divider()
# Tips & Tricks in the sidebar
st.sidebar.write("#### ⚙️ Vidokezo vya matumizi bora zaidi:")
tips = """
- Hakikisha uko katika **mazingira tulivu** ikiwa utarekodi sauti yako.
- Ongea **kwa uwazi na kwa kasi ya wastani** ukirekodi sauti yako.
- Ili kupakia faili yako uliyorekodi mapema, hakikisha ni **"mp3"** au **"wav"**.
"""
st.sidebar.markdown(tips)

# Give consent for the recording to be sent to a doctor/ hospital system
output_preference = st.radio("Je ungetaka kutuma suala lako la matibabu kwa daktari?", 
                             ["Ndio", "La"], index=1)

if output_preference == "Ndio":
    st.text_input("Ingiza namba yako ya simu")
else:
    pass

# Audio upload
#record_audio = st.radio("Je ungetaka kurekodi suala lako?", 
                            # ["Ndio", "La"], index=0)

record_audio = st.checkbox("Rekodi Sauti🎙️")
audio_ready = False
if record_audio==True:
#if record_audio=="Ndio":
        audio_file = audiorecorder("Bonyeza ili kurekodi⏺","Bonyeza ili kusitisha rekodi⏹") 
        #audio_file = audiorecorder("Bonyeza ili kurekodi⏺") 
        if len(audio_file) > 0:
            # To play audio in frontend:
            # st.audio(audio_file.export().read())  # audio_file.export().read() is a bytes object
            # To save audio_file to a file, use pydub export method:
            audio_file.export("audio_file.wav", format="wav")
            audio_ready = True
   
else:
    audio_file = st.file_uploader("Weka faili la sauti ulirokedi awali", type=['mp3', 'wav', 'ogg'])
    audio_ready = True 

# Convert audio to BytesIO
audio_file_to_bytes = False
if audio_file is not None:
    # Convert uploaded audio to wav
    if not record_audio:
        try:
            audio_file = AudioSegment.from_file(audio_file, format="mp3")
        except:
            audio_file = AudioSegment.from_file(audio_file, format="wav")
    # Convert audio to BytesIO
    audio_io = io.BytesIO()
    audio_file.export(audio_io, format="wav")
    st.audio(audio_io, format="audio/wav")
    #st.write("Sauti Imewekwa")
    audio_file_to_bytes = True
#else:
    #st.error("Tafadhali weka faili ya sauti")

# Transcribe
transcription=""
st.markdown("---")
if audio_ready:
    if st.button("Tafsiri sauti📝"):   
        with st.spinner('Hili ndilo swali lako...🤓'): 
            model = load_model()       
            result = model(audio_io.getvalue())['text']
            st.write('Hili ndilo swali lako:')
            transcription += result  # Use += for string concatenation
            st.success(result)

# Text generation -LLM
#st.markdown("---") 
#st.subheader('Subiri matokeo...') 
transcription2=transcription
information=""
if transcription2!="":
    with st.spinner("Subiri matokeo...🕒"):  
        transcription=get_prompt()+transcription #---if using gpt 3.5 uncomment
        #result2 = transcription_to_medical_tips(transcription) #chatgpt            
        result2=transcription
        information += result2  # Use += for string concatenation
        st.write('Haya ndio majibu:')
        st.success(information)
    
# Text to speech
if information!="":
    try: 
        st.markdown("---")
        st.write('Sikiliza majibu yako:') 
        text_to_voice(information, 'sw')
    except:
        pass
        #st.error("Tafadhali peana faili ya sauti")

    #write the code to save and download in a zip file
    transcript_txt=open("swali.txt","w")
    transcript_txt.write(transcription2)
    transcript_txt.close()

    information_txt=open("majibu.txt","w")
    information_txt.write(information)  
    information_txt.close()

    zip_file=ZipFile("matokeo.zip","w")
    zip_file.write("swali.txt")
    zip_file.write("majibu.txt")
    zip_file.close()

    with open("matokeo.zip", "rb") as f:
        btn=st.download_button(
            label="Pata matokeo📥",
            data=f,
            file_name="matokeo.zip",
            mime="application/zip",
        )


# conda install -c conda-forge ffmpeg
# streamlit run 🎙️_SemaAfya.py
