import os
import whisper
import streamlit as st
import openai 
import toml
from moviepy.editor import *
from fpdf import FPDF

import datetime
import sounddevice as sd
import soundfile as sf 

# Constants and variables
openai.api_key = toml.load('secrets.toml')['OPENAI_API_KEY']#os.getenv("OPENAI_API_KEY")
verbose=True

t = datetime.datetime.today()
D =  t.strftime('%d-%m-%y_%H%M%S')

WHISPER_MODEL='small'
WHISPER_LANG='eng'
audio_folder = './'

models_text=("Ada","Babbage","Curie","Davinci (3k + languages)")
models=['text-ada-001','text-babbage-001','text-curie-001','text-davinci-003']
models_dic=dict(zip(models_text,models))

languages=("English","Spanish","Hindi","Nepali")
languages_trans=("","in Spanish","in Hindi","in Nepali")
languages_dic=dict(zip(languages,languages_trans))

filename=""
sr = 44100
channels = 2

if "transcript" not in st.session_state:
    st.session_state['transcript'] = ""
if "summary" not in st.session_state:
    st.session_state['summary'] = ""
if 'saving' not in st.session_state:
    st.session_state['saving']=True
  
if "fileuploaded" not in st.session_state:
    st.session_state['fileuploaded'] = False
if "audio_file_raw" not in st.session_state:
    st.session_state['audio_file_raw'] = None

if "recording" not in st.session_state:
    st.session_state['recording'] = False
if "process_audio" not in st.session_state:
    st.session_state['process_audio'] = False
if "audio_file" not in st.session_state:
    st.session_state.audio_file=""

def save_pdf(): 
    pdf = FPDF() 
    pdf.add_page()
    if language_radio in ["Hindi","Nepali"]:
        pdf.add_font('Mangal', '', './fonts/Mangal.ttf', uni=True)
        pdf.add_font('TiroDevanagariHindi', '', './fonts/TiroDevanagariHindi.ttf', uni=True)
        font="Mangal"
        #!/usr/bin/env python
        # -*- coding: utf32 -*-
    else:
        font="Arial"

    pdf.set_font(font,"B", size = 20)
    pdf.cell(0, 10, txt = f"Transcript - {D}", align = 'C')
    pdf.ln()
    pdf.set_font(font,"I", size = 12)
    pdf.multi_cell(0, 10, txt = "Summary:\n"+st.session_state['summary'])
    pdf.ln()
    pdf.set_font(font, size = 15)
    pdf.multi_cell(0, 10, txt = "Transcript:\n"+ st.session_state['transcript'])
    pdf.ln()
    pdf.output(f"transcript_{D}.pdf").encode('latin-1','ignore') 
    return ""
 

def get_summary(prompt,model="text-ada-001",language="", verbose=True):
    tokens=int(1000) if int(len(prompt)/4)>250 else int(len(prompt)/4)
    augmented_prompt = f"summarize this text {language}: {prompt}"
    print(f"sumarize({model},{language})>>>>>",augmented_prompt)
    try:
        with st.spinner("Summarizing..."):
            st.session_state['summary'] = openai.Completion.create( 
                model =  model,  
                prompt = augmented_prompt,
                temperature=.5,
                max_tokens= tokens,
            )['choices'][0]['text'].strip()
            st.session_state['saving']=False

    except Exception as e:
        error="There was an error", str(e) if verbose else ""
        print(error)
        st.session_state['summary'] = error
        st.session_state['saving']=True
     

##################
######LAYOUT######
##################
st.title("Lecture Summarizer")
st.subheader("Upload a lecture, and get a summary")


st.text("Select this first")
model_radio=st.radio("Model",models_text, horizontal=True)
language_radio="English"
if(model_radio == models_text[3]):
    language_radio=st.radio("Translate to:",languages, horizontal=True)

#selected model
_model= models_dic[model_radio]
_language=languages_dic[language_radio]

st.text("Record lecture or upload an mp3 file to transcript:")

##################
# Record audio
##################
def record_audio( duration):
    filename = f"recording_{D}.wav"
    # Record audio for the specified duration
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels)
    sd.wait()
    # Save the recording to disk
    sf.write(filename, recording, sr)    
    print(f"Recording saved to {filename}")
    st.session_state['process_audio'] = True
    st.session_state.audio_file=filename
    
duration =  st.slider("Recording duration (seconds)", min_value=1, max_value=300, value=10)
if st.button("Record"):
    with st.spinner("Recording..."):
        record_audio(duration)
            
            
##################
## Select an mp3 file
##################
if not os.path.exists(filename):
    audio_file_raw = st.file_uploader("Choose a file", type=['mp3','m4a','wav'], key="fileupload",disabled=st.session_state.fileuploaded )
    if audio_file_raw is not None:
        st.session_state.audio_file=audio_file_raw.name
        filename=audio_file_raw.name
        st.session_state['process_audio'] = True #entra en loop, porque la condicion se cumple de nuevo

##################
## Process audio
##################

if st.session_state.process_audio:
    with st.spinner("Getting Transcript"):
        audio_file=st.session_state.audio_file
        output_folder=os.path.join('output','output-'+audio_file[:4])
        output_file=f'transcript_{WHISPER_LANG}_{WHISPER_MODEL}_'+audio_file[:-4]
        path = os.path.join(audio_folder,audio_file)
            
        # create output folder if doesn't exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    with st.spinner('Wait for it...'):
        #Using whisper
        try:
            model = whisper.load_model(WHISPER_MODEL, device='gpu')
            print("GPU Found,")
        except:
            print("No GPU found, using CPU")
            model = whisper.load_model(WHISPER_MODEL, device='cpu')
            
        def save_txt_file(list_txt):
            path_to_save=os.path.join(output_folder,output_file)
            with open(path_to_save, mode="a+") as f: 
                f.write(list_txt+'\n') 
        
        final_list_of_text = []
        total_text=[]
        audio_file=path
        out = model.transcribe(audio_file)

        st.session_state['fileuploaded'] = True
        save_txt_file(out['text'])
        st.session_state['transcript'] = out['text']  

        get_summary(out['text'],_model,_language)
        #Avoid repeating the process once it is done
        st.session_state.process_audio=False

 
transcript=st.text_area("Transcript",st.session_state['transcript'])
summary_text=st.text_area("Summary",st.session_state['summary'])
save_btn=st.button("Save PDF",on_click=save_pdf, disabled=st.session_state['saving'])