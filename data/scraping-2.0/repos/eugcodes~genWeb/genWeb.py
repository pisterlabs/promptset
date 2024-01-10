import openai
import streamlit as st

import datetime

import os

openai.api_key = st.secrets["apiSecret"]

# Create initial pearl
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
      {"role": "system", "content": "You are wise, loving, positive, and warm. Your role is to offer concise pearls of wisdom. You write in the style of Jenny Slate and David Sedaris. Your responses are up to 3 sentences long. Provide wisdom on one topic only. Be quirky, whimsical, funny, and endearing."},
      {"role": "user", "content": "Offer some concise wisdom on a random important topic related to living well and wisely. Avoid using ther phrase: on the topic of"}
  ],
  temperature=0.9,
  max_tokens=128,
)

# Write text
st.title("Welcome, digital wanderer!")
st.write("This is an experimental sandbox.")
st.write("\nDaily nugget of wisdom:")

# Write first pearl
st.write(response["choices"][0]["message"]["content"])

# create and initiatize app hit counter
if 'hitCount' not in st.session_state:
    st.session_state.hitCount = 0

# increment app hit count
#st.session_state.hitCount += 1

increment = st.button('Please sir, I want some more!')

if increment:
    st.session_state.hitCount += 1

st.write("\n", st.session_state.hitCount,"served")


# improvements
# 1. improve performance. Cache results in queue?
# 2. perist counter across all sessions. e.g. write/retrieve count to static file?



