import streamlit as st
import openai
import json

user_api_key = st.sidebar.text_input("OpenAI API key", type="password")
client = openai.OpenAI(api_key=user_api_key)

#client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as a music expert who is passionate about the artist. my client want to know the artist's most popular song.
give 5 of the most songs of target artist that client should listen to. and give the description of suggested song 
-when the song released and the artist's age at that time
-the artist's living condition and the bad situation that he/her is facing on that time.
-why the song was written
-what is this the meaning of the song.
all of the answers should be long.
        """    

st.title("Tell me :rainbow[who is your favorite artist!]")

st.markdown("input the name of your favorite artist.  \n\
            i will tell you what song you should listen to, and tell you some specific details of the songs. \n\
            ")

user_input = st.text_input("Input the name here")
# st.button("Submit")


if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    st.markdown('**AI response:**')
    answer = response.choices[0].message.content
    st.write(answer)

