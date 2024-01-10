import streamlit as st
from io import BytesIO
import requests,json
import streamlit.components.v1 as components
from st_custom_components import st_audiorec
from audio_recorder_streamlit import audio_recorder
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import openai,spacy
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

#open ai key is in config.json
with open('./config.json','r') as f:
    data = f.read()
    config = json.loads(data)
    openai.api_key = config['openai_key']
# Set page title



def ask_gpt(question):
    e =openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"{question} give positive reinforcment to the user! talk alittle about the current scores"}])
    return e['choices'][0]['message']['content']

st.set_page_config(page_title="Aura")
SERVER_URL = "http://127.0.0.1:8080/upload_video"
# Define sidebar
st.sidebar.title("Navigation")
menu = ["🎥 Recorder", "📚 History", "📁 Archieve", "🔍 Analysis"]
choice = st.sidebar.selectbox("Go to", menu)

# Add emojis for each section
if choice == "🎥 Recorder":
    st.title("🎤 Audio recorder")
    wav_audio_data = st_audiorec()
 
    if wav_audio_data is not None:
        # Display the recorded audio in the Streamlit app
        st.audio(wav_audio_data, format='audio/wav')
        
        # If the Submit button is clicked, send the audio data to the server
        if st.button('Check!'):
            # You need to convert the audio data from the Bytes datatype to a File-like object that can be handled by the requests library
            with st.status("Uploading File...", expanded=True) as status:
                audio_file = BytesIO(wav_audio_data)
                files = {'audio_data': audio_file}
                # Make a POST request to the server with the recorded audio file
                res = requests.post(SERVER_URL, files=files)
                print(res.json())
                if res.ok:
                    st.success("Audio successfully uploaded to server.")
                else:
                    st.error("Error occurred while uploading audio to server.")
                st.info("Analyzing Emotion...")
                resp = ask_gpt(str(res.json()))
                st.success("Response Ready!")
                status.update(label="File complete!", state="complete", expanded=False)
            st.success(resp)

elif choice == "📚 History":
    col1, col2, col3 = st.columns(3)
    negative_Data = [0.2, 0.3, 0.1, 0.4, 0.3, 0.2, 0.1, 0.2]
    neutral_data = [0.5, 0.4, 0.6, 0.3, 0.4, 0.5, 0.6, 0.5]
    positive_data = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    #find average of all of the data
    average_negative = np.average(negative_Data)
    average_neutral = np.average(neutral_data)
    average_positive = np.average(positive_data)
    col1.metric("Positivity", f"{average_positive*100}%", "7%")
    col2.metric("Neutral",f"{average_neutral*100}%", "-10%")
    col3.metric("Average Negativity", f"{average_negative*100}%", "4%",delta_color="inverse")

    data = {
        'time': ['12:00 AM', '3:00 AM', '6:00 AM', '9:00 AM', '12:00 PM', '3:00 PM', '6:00 PM', '9:00 PM'],
        'negative': negative_Data,
        'neutral': neutral_data,
        'positive': positive_data
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Create the chart
    fig = px.line(df, x='time', y=['negative', 'neutral', 'positive'], title='Emotions for the Current Day')

    

    # Define sample data
    data = {
        'date': pd.date_range(start='2022-01-01', end='2022-01-07'),
        'negative': [0.2, 0.3, 0.1, 0.4, 0.3, 0.2, 0.1],
        'neutral': [0.5, 0.4, 0.6, 0.3, 0.4, 0.5, 0.6],
        'positive': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Create the chart
    fig = px.line(df, x='date', y=['negative', 'neutral', 'positive'], title='Feelings over a Week')

    # Add a button to switch to monthly view
    if st.button('Switch to Monthly View'):
        # Define sample data for monthly view
        data_monthly = {
            'date': pd.date_range(start='2022-01-01', end='2022-01-31'),
            'negative': [0.2, 0.3, 0.1, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1],
            'neutral': [0.5, 0.4, 0.6, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.6, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.6, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.6, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.6],
            'positive': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,0.3]
        }

        # Create a DataFrame from the data
        df_monthly = pd.DataFrame(data_monthly)

        # Create the chart for monthly view
        fig_monthly = px.line(df_monthly, x='date', y=['negative', 'neutral', 'positive'], title='Feelings over a Month')

        # Display the chart for monthly view
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        # Display the chart for weekly view
        st.plotly_chart(fig, use_container_width=True)
elif choice == "📁 Archieve":
    st.title("📁 Archieve")
    st.write("Listen in on your previous conversations!")
    r = requests.get("http://127.0.0.1:8080/archieve")
    rjson = r.json()
    for i in range(len(rjson['videos'])):
        vid = rjson['videos'][i]
        timestamp = rjson['timestamp'][i]
        vid2 = vid.split("-")[0]
        if st.button(timestamp+" | ID:  "+vid2):
            # Do something when the button is pressed
            print(f"The button for video {vid} was pressed.")
            r = requests.get(f"http://127.0.0.1:8080/view/{vid}")
            #display the video
            st.video(r.content)
            #add a close button
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                if st.button('Close'):
                    st.stop()
            with col2:
                if st.button('Analysis'):
                    print("hi")
                    r = requests.get(f"http://127.0.0.1:8080/reanalysis",json={'id':vid})
                    print(r.json())
            with col3:
                st.button('Delete')
elif choice == "🔍 Analysis":
    st.title("🔍 Analysis")


    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = [
        {"role": "assistant", "content": "Hello there! I noticed a significant increase in your anxiety levels last Tuesday. Would you like to discuss what happened or anything that you may have been experiencing that day?"}  # add this line    
        ]
    data = {
    'date': pd.date_range(start='2022-01-01', end='2022-01-07'),
    'negative': [0.2, 0.3, 0.1, 0.4, 0.8, 0.2, 0.1], # Added a spike on Tuesday
}

# Create a DataFrame from the data
    df = pd.DataFrame(data)

# Create the chart
    fig = px.line(df, x='date', y='negative', title='Negativity Levels for the Week')

# Display the chart
    st.plotly_chart(fig, use_container_width=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


else:
    st.title("🏠 Home") 