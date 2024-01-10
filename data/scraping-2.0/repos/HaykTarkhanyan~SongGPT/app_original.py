from random import choice
import streamlit as st
import logging 
import openai
from utils import clean_output
import os 

openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(filename='app.log', filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)

# favorite_songs = ['Lake of fire by Nirvana', 'Aerials by SOAD']
# default_song = choice(favorite_songs)

st.title("Your favorite songs about your topic")

logging.info("App started")

topic = st.text_input("Enter your topic here")
# TODO change the mutiselect to a dropdown
song = st.multiselect("Select the song", ["Bohemian Rhapsody", "Hotel california", "Real slim shady", "Դրախտի ալվան ծաղիկ", 'Lake of fire by Nirvana', 'Aerials by SOAD'], 
                      default="Real slim shady")
seriousness = st.slider("How serious is your topic?", 0, 2)

seriousness_dict = {0: "serious", 1: "funny", 2: "very funny",}

seriousness = seriousness_dict[seriousness]

if st.button("Submit"):
    logging.info(f"User submitted {topic} {song} and {seriousness}")
    if not topic:
        logging.warning("User did not enter a topic")
        st.write("Please enter your topic")
    else:    
        prompt = f"Please change the text of the song `{song[0]}` to be about `{topic}`, make it {seriousness}, \
                   keep the song structure and the rhymes. Provide your output in the following format \
                   <Song name> \
                   <Original lyrics> \
                   <Changed lyrics>"
        st.write(f"prompt: {prompt}")\
        
        # TODO postprocess the output
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                       messages=[{"role": "user", "content": prompt}])
        response = chat_completion.choices[0].message.content
        
        st.subheader("Response")
        st.write(response)
        # st.subheader("Cleaned output (might be wrong)")
        # st.write(clean_output(response))
        
        logging.info(f"Response: {response}")
        st.balloons()
        st.snow()