import streamlit as st
from pymongo import MongoClient
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import os
import openai
from datetime import datetime
import pymongo
from helper import *


os.environ['OPENAI_API_KEY'] = ''

openai.api_key = os.getenv('OPENAI_API_KEY')

base = os.getcwd()

client = MongoClient('mongodb://localhost:27017')

db = client['lumoscribe']

collection = db['records']

# Set page config
st.set_page_config(
    page_title="LumoScribe",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for lottie
@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Add additional CSS styles
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

# Check for query parameters to set the selected page
query_params = st.experimental_get_query_params()

# Sidebar for navigation
with st.sidebar:
    st.image(f"images/logo.png", use_column_width=True)
    # Determine the selection based on the query parameters
    selected = query_params.get("selected", ["Transcribe Lectures"])[0]
    # Use the selection to set the default index for the sidebar menu
    selected = option_menu('LumoScribe', ["Transcribe Lectures", 'About Us', 'Previous Lectures'], 
        icons=['play-btn', 'info-circle', 'intersect'],
        menu_icon="cast", default_index=["Transcribe Lectures", 'About Us', 'Previous Lectures'].index(selected))
    
    lottie = load_lottiefile(f"{base}\\similo3.json")
    st_lottie(lottie, key='loc')

# Pages
if selected == "Previous Lectures":
    st.title('Previous Lectures')
    data = get_data(collection)
    if 'audio_id' in query_params:
        audio_id = query_params['audio_id'][0]
        row = data.loc[data['_id'] == audio_id].iloc[0]
        show_audio_details(row)
    else:
        if not data.empty:
            data['link'] = data.apply(lambda row: f"<a href='/?selected=Previous%20Lectures&audio_id={row['_id']}'>View Details</a>", axis=1)
            st.write(data[['created', 'title', 'duration', 'subject', 'link']].to_html(escape=False), unsafe_allow_html=True)

elif selected == "Transcribe Lectures":
    st.title("Transcribe Your Lectures - Summarize, Test, Repeat")

    lecture_title = st.text_input("Enter title for transcription")

    subject = st.selectbox(
                        "Lecture Subject",
                        ("AIPI520", "MENG540", "EGRMGMT130", "POLSCI430"),
                        index=None,
                        placeholder="Select Subject",
                        )

    uploaded_file = st.file_uploader("Choose an MP3 or MP4 file", type=["mp3", "mp4"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the file details
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "DateOfUpload": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Use current time as upload date
        }

        if 'api_results' not in st.session_state:
            make_api_call(f"{base}\\{uploaded_file.name}")

        record = (st.session_state.api_results)

        if 'action_executed' not in st.session_state:
            st.session_state.action_executed = False

        if not st.session_state.action_executed:
            # Perform the action, e.g., an API call
            record['audio_path'] = f"{base}\\{uploaded_file.name}"
            record['title'] = lecture_title
            record['subject'] = subject
            collection.insert_one(record)
            
            st.session_state.action_executed = True

        st.success('File uploaded and transcribed successfully!')

        latest_record = collection.find_one(sort=[("_id", pymongo.DESCENDING)])

        gen_navbar(latest_record)

elif selected == "About Us":
    st.title('About Us')
    team_members = [
    {
        "name": "Suneel",
        "description": "Degree",
        "image_path": f"images/suneel.png" 
    },
    {
        "name": "Rucha",
        "description": "Degree",
        "image_path": f"images/rucha.png"
    },
    {
        "name": "Aafra",
        "description": "Degree",
        "image_path": f"images/Aafra.png"
    },
    {
        "name": "Sri",
        "description": "Degree",
        "image_path": f"images/sri.png"
    }
    ]

    display_team(team_members)
