import streamlit as st
import pandas as pd
from pytube import YouTube
import requests
import time
import assemblyai as aai
from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from openai.error import AuthenticationError
from langsmith import Client

from llm_stuff import (
    _DEFAULT_SYSTEM_PROMPT,
    get_memory,
    get_llm_chain,
    StreamHandler,
    get_langsmith_client,
)

# auth_key from secrets
auth_key = st.secrets['auth_key']

# global variables
audio_location = ''
audio_url = ''
link = ''
link_new = ''

# Initialize State
if "trace_link" not in st.session_state:
    st.session_state.trace_link = None
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ''
st.session_state.transcription = ''
st.session_state.process_status = ''    

st.session_state.chat_text = ''    
    
# youtube-dl options
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': './%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
CHUNK_SIZE = 5242880

# endpoints
upload_endpoint = 'https://api.assemblyai.com/v2/upload'

headers = {
    "authorization": auth_key,
    "content-type": "application/json"    
}

st.set_page_config(page_title='LLM with Streamlit', 
                   page_icon='👀', layout='centered', initial_sidebar_state='expanded' )

# to hide streamlit menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
# pass javascript to hide streamlit menu
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# @st.cache_data(ttl=600)
def download_audio(link):
    global audio_location
    _id = link.strip()
    
    def get_vid(_id):
        # create object YouTube
        yt = YouTube(_id)
        audio_stream = yt.streams.filter(only_audio=True).first()
        # print(audio_stream)
        return audio_stream
        
    # download the audio of the Youtube video locally
    audio_stream = get_vid(_id)
    download_path = './'
    audio_location = audio_stream.download(output_path=download_path)   
    # print('Saved audio to', audio_location)
    
def read_file(filename):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(CHUNK_SIZE)
            if not data:
                break
            yield data
            
def upload_audio():
    global audio_location
    global audio_url
    upload_response = requests.post(
        upload_endpoint, 
        headers=headers, 
        data=read_file(audio_location)
    )
    
    audio_url = upload_response.json()['upload_url']


def gauge_chart(value, max_value, label):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Define angules
    start_angle = 0
    end_angle_red = 180
    end_angle_green = 180 - (value / max_value) * 180 # reverte start point
    
    # arc widgth and radius
    arc_width = 0.2  # width of the arc
    arc_radius = 0.4  # Radius of the arc

    # Arc green
    arc_green = Arc((0.5, 0.5), arc_radius * 2, arc_radius * 2, angle=0, theta1=start_angle, theta2=end_angle_red, color='green', lw=40)
    ax.add_patch(arc_green)

    # Arc red
    arc_red = Arc((0.5, 0.5), arc_radius * 2, arc_radius * 2, angle=0, theta1=start_angle, theta2=end_angle_green, color='red', lw=40)
    ax.add_patch(arc_red)

    # aditional settings
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # explain text
    ax.text(0.5, 0.6, "{:.1f}%".format(round(value, 1)), ha='center', va='center', fontsize=20)
    ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=16)
    ax.text(0.5, 0.25, "Global Confidence", ha='center', va='center', fontsize=26, color='black')
    ax.text(0.5, 0.1, "Greater green bar is better", ha='center', va='center', fontsize=18, color='green')
    return fig

def main():
    global audio_location
    global audio_url
    global link
    global link_new
    
    with st.sidebar:
        # st.image('logo-250-transparente.png')
        st.header('Information')
        st.write("""
                This project was created with the goal of participating in the 'Streamlit LLM Hackathon 2023'. 
                \nThis site uses **AssemblyAI** to transcribe audio from YouTube videos and **LangChain** to handle chat.
                \nTo chat about the video, please, supply your OPENAI API KEY.
                \nAt this point, the video must be in English.
                """)
        st.header('OpenAI API KEY')
        if st.session_state.openai_api_key == '':
            st.write("""
                    ❗ If you want to "ask questions about" the video, please, supply your OPENAI API KEY **before** starting.
                    """)
            st.session_state.openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        else:
            st.write('Using the OPENAI API KEY already supplied before.')
        
        st.header('About')
        st.write('Details about this project can be found in https://github.com/htsnet/StreamlitHackathonLLM2023')
    # título
    title = f'Audio transcription and analysis with LLM'
    st.title(title)
    
    subTitle = f'Using a Youtube video link, this site will transcribe the audio and show relevant information.'
    st.subheader(subTitle)

    # information tabs  
    st.markdown('<style>[id^="tabs-bui3-tab-"] > div > p{font-size:20px;}</style>', unsafe_allow_html=True)
    # emoji list https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
    tab1, tab2, tab3, tab4 = st.tabs(['📹:red[ **Video Process**]', '📖:red[ **Transcription**]', '📄:red[ **Sumary**]', '🏷️:red[ **Categories**]'])

    with tab1:
        st.subheader('Start here!')
        
        # link
        link = st.text_input('Paste your Youtube video link and press Enter')
        
        # download stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))        
        
        if link != '':
            if link_new == link:
                st.toast('Video already processed!', icon='❗')
            else:
                link_new = link
                time_start = time.time()
                aai.settings.api_key = auth_key
                
                col1, col2, col3 = st.columns(3)
                # using col1 to reduce width of video
                with col1:
                    st.video(link)

                try: 
                    with st.spinner('Getting audio... (1/3)'):
                        download_audio(link)
                        
                    with st.spinner('Uploading audio... (2/3)'):
                        upload_audio()  
                        
                    with col2:
                        st.write('Uploaded audio to', audio_url)  
                    
                    with st.spinner('Transcribing audio... (3/3)'):
                        config = aai.TranscriptionConfig(
                            speaker_labels=True, 
                            iab_categories=True
                            )

                        transcriber = aai.Transcriber()
                        transcript = transcriber.transcribe(
                        audio_url,
                        config=config
                        )
                        
                        # st.markdown(transcript.text)
                        st.session_state.transcription = transcript
                        
                        # dictionary to store the words
                        word_counts = defaultdict(int)
                        
                        # regular expression to remove punctuation
                        word_pattern = re.compile(r'\b\w+\b')
                        word_count = 0
                        confidence_values = 0
                        
                        if st.session_state.transcription.error:
                            st.write(st.session_state.transcription.error)
                            st.session_state.process_status = 'error'
                            st.toast('Problem with the video!', icon='❗')
                            st.stop()

                        try:
                            # read json result and count words
                            for pieces in transcript.utterances:
                                words = pieces.words
                                for word in words:
                                    # remove punctuation and convert to lowercase
                                    text = word_pattern.findall(word.text)
                                    # sum 1 for each word found, if not empty
                                    if text and text[0] not in stop_words:
                                        word_counts[text[0].lower()] += 1
                                        word_count += 1
                                        confidence_values += word.confidence
                        except Exception as e:
                            st.write(e)

                        st.session_state.process_status = 'done'
                        if st.session_state.openai_api_key != '':
                            st.write("Ask question about the content! **Click on sidebar** to go to chat.")
                        else:
                            st.write("Sorry, there wasn't an OPENAI API KEY to ask questions...")
                        time_stop = time.time()
                
                except Exception as e:
                    st.write('Error! Maybe the video is private. Try another')
                    st.write(e)
                    st.session_state.process_status = ''
                    time_stop = time.time()
                    st.toast('Problem with the video!', icon='❗')
                    st.stop()  
                    
                with col2:
                    time_total = time_stop - time_start
                    st.write('🕔 Processed in',  "{:.1f}".format(round(time_total, 1)), 'seconds!')
                    
                with col3:
                    # st.markdown(f"Total words: {word_count}")
                    # st.markdown(f"Total confidence: {confidence_values}")
                    # st.markdown(f"Average confidence: {confidence_values/word_count}")
                    if word_count > 0:
                        confidence = confidence_values/word_count * 100
                    else:
                        confidence = 0
                    
                    # Gauge Chart
                    max_value = 100
                    st.pyplot(gauge_chart(confidence, max_value, f'{word_count} words'))
                    
                st.markdown('See the tabs above for information about the audio!')
                st.toast('Great. Video processed! Enjoy', icon='🎉')

    with tab2:
        st.subheader('Audio Transcription')
        if st.session_state.process_status == 'done':
            # Get the parts of the transcript that were tagged with topics
            st.session_state.chat_text = ''
            for result in st.session_state.transcription.iab_categories.results:
                st.session_state.chat_text += result.text + ' '
                st.markdown(result.text)
                # st.markdown(f"Timestamp: {result.timestamp.start} - {result.timestamp.end}")
                # for label in result.labels:
                #     st.markdown(label.label)  # topic
                #     st.markdown(label.relevance)  # how relevant the label is for the portion of text
        else:
            st.markdown('Process the video first!')

    with tab3:
        st.subheader('Sumary')
        if st.session_state.process_status == 'done':
            # sort descending
            sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))

            # show the words more used
            st.write("WORDS USED MORE THAN 3 TIMES")
            word_count_tuples = [(word, count) for word, count in sorted_word_counts.items() if count > 3]

            # create a dataframe with the list of tuples
            df = pd.DataFrame(word_count_tuples, columns=["Word", "Count"])

            # show the dataframe            
            st.table(df)
        else:
            st.markdown('Process the video first!')
    
    with tab4:
        st.subheader('Relevant Categories')
        if st.session_state.process_status == 'done':
            # Get a summary of all topics in the transcript
            for label, relevance in st.session_state.transcription.iab_categories.summary.items():
                relevance = "{:.1f}%".format(round(relevance * 100, 1))
                st.markdown(f"{label} ({relevance})")
        else:
            st.markdown('Process the video first!')
            
if __name__ == '__main__':
	main()   