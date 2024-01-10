
'''
Streamlit function
'''

# 1. Import packages
# 1.1 Python packages

import streamlit as st
import numpy as np
import pandas as pd
import time
import openai
import matplotlib.pyplot as plt
import requests


# 1.2 Model Tasks
from main_2 import main # UPDATE TO MAIN

# 1.2 Input dictionaries
composer_info_dict = {
                        'Bethel Music': ' Formed in 2001, Bethel Music emerged from Bethel Church in California, specializes in contemporary Christian and worship music. Their style is characterized by modern, dynamic worship songs that blend pop and rock elements with religious themes.',
                        'Richard Clayderman' : 'French pianist Richard Clayderman (born 1953), known for his melodic and romantic style, is a leading figure in new age and pop-classical music. His renditions often feature a blend of popular songs, movie soundtracks, and classical pieces.',
                        'Ludovico Einaudy': 'Italian pianist and composer Ludovico Einaudi (born 1955) is renowned for his contemporary classical music, which incorporates elements of minimalism, ambient, and pop music. His compositions are known for their emotional depth and simplicity.',
                        'Herbie Hancock' : 'American musician Herbie Hancock (born 1940) is a legendary figure in jazz, particularly known for his contributions to jazz fusion. His work incorporates elements of funk, soul, and electronic music, showcasing his innovative approach to jazz.' ,
                        'Hillsong Worship': 'Hillsong Worship, from Australia (formed 1983), is synonymous with contemporary Christian music. Their style combines Christian worship with contemporary music elements, influencing the genre globally with their modern worship anthems.' ,
                        'Joe Hisaishi' : 'Japanese composer Joe Hisaishi (born 1950), is celebrated for his film scores, particularly for Studio Ghibli. His style, while primarily contemporary, seamlessly blends orchestral and minimalist elements, creating emotionally resonant soundtracks.' ,
                        'Ryuichy Sakamoto': 'Ryuichi Sakamoto (born 1952), initially part of Yellow Magic Orchestra, has a diverse musical style encompassing electronic, experimental, and classical music. His innovative compositions often feature a mix of traditional instruments and electronic sounds.',
                        'Yiruma': 'South Korean pianist Yiruma (born 1978) is known for his contemporary classical music that often crosses into the pop genre. His compositions, typically for solo piano, are characterized by their emotive melodies and accessible style.'
}

composer_genre_dict = { 'Bethel Music': 'Religious',
                        'Richard Clayderman' : 'New Age/Pop-Classical',
                        'Ludovico Einaudy': 'Contemporary Classical',
                        'Herbie Hancock' : 'Jazz' ,
                        'Hillsong Worship': 'Contemporary Christian' ,
                        'Joe Hisaishi' : 'Film Music/Contemporary' ,
                        'Ryuichy Sakamoto': 'Electronic/Experimental',
                        'Yiruma': 'Contemporary Classical/Pop'}

composer_image_dict = {'Bethel Music': 'resources/composer_images/bethel_church.png',
                        'Richard Clayderman' : 'resources/composer_images/clayderman.png',
                        'Ludovico Einaudy': 'resources/composer_images/einaudi.png',
                        'Herbie Hancock' : 'resources/composer_images/herbie_hancock.png' ,
                        'Hillsong Worship': 'resources/composer_images/hillsong_worship.png' ,
                        'Joe Hisaishi' : 'resources/composer_images/joe_hisaishi.png' ,
                        'Ryuichy Sakamoto': 'resources/composer_images/ryuichi_sakamoto.png',
                        'Yiruma': 'resources/composer_images/yiruma.png'}

example_prediction = {  'Bethel Music': 0.8,
                        'Richard Clayderman' : 0.1,
                        'Ludovico Einaudi': 0.1}


# 2. Website
st.set_page_config(
    page_title="TuneScout",
    page_icon="🎵",
    #layout="wide",
    initial_sidebar_state="collapsed"
)

# 2.2 Website Layout: banner & header
st.markdown("""
<style>
.banner {
  background: #000000;
  color: white;
  padding: 10px;
  position: relative;
  text-align: center;
  border-radius: 10px;
  max-height: 400px;  /* Banner max height */
  overflow: hidden;  /* Hide excess banner image */
}
.banner h1 {
  color: white; /* White color for TuneScout */
  border-bottom: 2px solid #68D3CE; /* Line under TuneScout with color #68D3CE */
  padding-bottom: 5px; /* Add some space between text and line */
}
</style>

<div class="banner">
  <h1>TuneScout</h1>
  <p><br>For the audiophiles,<br>the musical nomads,<br>for the ones who never heard enough</p>
</div>
""", unsafe_allow_html=True)

if 'mp3_response' not in st.session_state:
    st.session_state['mp3_response'] = None

if 'artist_response' not in st.session_state:
    st.session_state['artist_response'] = None

# 2.3 Load MP3
input_file = st.file_uploader('',type=["mp3"]) # The st.file_uploader function returns a BytesIO object for the uploaded file.


if input_file is not None:
    if st.session_state['mp3_response'] is None:
        with st.spinner('Dusting off the records...'):
            url = "https://music-fbzdapc47q-ew.a.run.app"
            files = {'file': input_file}
            response = requests.post(url,files=files).json()
            st.session_state['mp3_response'] = response
    else:
            response = st.session_state['mp3_response']

    composers = sorted(response.keys())
    if "composer" not in st.session_state:
        st.session_state["composer"] = 0
    selected_composer = st.selectbox('Select a artist:', composers, st.session_state["composer"] )
    st.write(f"You selected: {selected_composer}")

    # 2.4 Buttons
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)


    # Button 1: Suggest
    if col1.button('Curate'):
        with st.spinner('Matching your tune to artists in our database...'):
            st.write("Dusted off the records, here are some artists that match your taste..")
            st.markdown("""
                <div style='text-align: center;'>
                    <h2>Curated Artists</h2>
                </div>
            """, unsafe_allow_html=True)

            bar_colors = ['#5CCBD0', '#3BB5DA', '#5CD9C7']  # Adjust the colors as needed

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(response.keys(), response.values(), color=bar_colors)

            # Set the color of x-axis ticks and tick lines
            ax.tick_params(axis='x', colors='#00393F')
            ax.set_xticklabels(response.keys(), fontsize=15)
            ax.xaxis.label.set_size(14)
            ax.tick_params(axis='y', colors='#00393F')

            # Set the color of y-axis grid lines
            ax.yaxis.grid(color='#E6F4F1', linestyle='dashed', alpha=.8)

            # Remove y-axis ticks
            ax.tick_params(axis='both', which='both', bottom=False, left=False, top=False)

            # Hide the top, right, and left spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Set the color of the x-axis line
            ax.spines['bottom'].set_color('#00393F')


            #ax.set_title('Curated Artists')
            ax.set_yticks([])
            #ax.set_yticklabels(['Different Styles', 'Pretty Congruent', 'Similar Tune'], fontsize=15, ha='left', rotation=35)
            ax.yaxis.labelpad = 10
            ax.tick_params(axis='y', pad=30)
            #ax.set_xlabel('Artist')
            ax.set_ylabel('Similarity', fontsize=14)


            # Show the plot in the sidebar
            st.pyplot(fig)

    # Button 2: Inspire me
    if col2.button('Discover'):
        with st.spinner('Exploring the cyberspace for some musical gems...'):
            time.sleep(5)
            st.write("Finding music to get you inspired")
            api_key = st.secrets["openai"]["api_key"]
            client = openai.OpenAI(api_key=api_key)
            #selected_composer = max(response.items(), key = lambda x: x[1])[0]
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
            {
                "role": "system",
                "content": "You are an informative and helpful assistant, skilled in explaining recommendations for selected_composers in music."
            },
            {
                "role": "user",
                "content": "We already have a description like this: Bethel Music (Religious): Formed in 2001, Bethel Music emerged from the Bethel Church in Redding, California. [...] Yiruma (Born 1978): South Korean pianist and selected_composer Yiruma (Lee Ru-ma) gained international fame in the early 2000s. His melodic and accessible compositions, often falling into the contemporary classical and pop genres, have made him popular among diverse audiences."
            },
            {
                "role": "user",
                "content": f"We have a selected_composer that was predicted by a model which is {selected_composer}. \
                                We want you to give recommendation similar artists - Artist Name. \
                                We want 3 song recomendation for each artist. \
                                No description of {selected_composer} but rather only the ones that are similar and why. \
                                we dont need the a introduction on the {selected_composer}.\
                                Please provide the information in the following format:\
                                \n\n1. Artist Name\
                                \n   - Reason for similarity:\
                                \n   - Song recomendation for Artist Name:\
                                \n\nRepeat this for four artists."\
            }
            ]
            )
            for message in completion.choices:
                st.write(completion.choices[0].message.content)
                st.markdown("***")



    # Button 3: Info
    if col3.button('Insights'):
        if True:
            selected_composer = max(response.items(), key=lambda x: x[1])[0]

            #artist title
            st.markdown(f"<h1 style='text-align: center'>{selected_composer} - {composer_genre_dict[selected_composer]}</h1>", unsafe_allow_html=True)
            #artist image
            composer_image = composer_image_dict[selected_composer]
            st.image(composer_image, use_column_width=True)
            #artist info text
            st.markdown(f"<div style='text-align: justify'>{composer_info_dict[selected_composer]}</div>", unsafe_allow_html=True)
