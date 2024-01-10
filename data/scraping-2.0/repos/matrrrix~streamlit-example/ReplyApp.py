from collections import namedtuple
import streamlit as st
import openai

"""
# Canvas Discussion Reply Generation!

This is a project made for the [AI Club](https://aiclub.sdsu.edu/) at SDSU.

The purpose of this project is for research purposes to test the capabilities of natural language processing. Neither the project members nor the AI Club are responsible for malevolent usage.

The model utilized within this project comes a fine tuned version of the base Curie model by OpenAI, meaning that the usage pricing is **\$0.0120 / 1K tokens**. Usage of the model in this app is tied directly to my OpenAI account, so I have put a hard limit of **\$0.5** of usage per month. If an error occurs when attempting to generate a reponse, this likely means that this monthly limit has been reached.

"""

# user_key = st.text_input('Provide your OpenAI API Key', type="password")

user_prompt = st.text_input('What is the Canvas Discussion Prompt?')

token_number = st.slider('Select the number of tokens that you wish to generate (think word count). Read above for pricing.', min_value=50, max_value=500, value=150, step=1)

if st.button("Generate", disabled=((user_prompt is None) or (token_number is None))):
    openai.api_key = st.secrets["api_key"]

    user_prompt = user_prompt + " ->"

    # old mode with less fine-tune data curie:ft-personal-2023-03-12-09-46-53
    reply = openai.Completion.create(
    model="curie:ft-personal-2023-04-06-22-16-08",
    prompt= user_prompt,
    max_tokens=token_number)

    modified = reply['choices'][0]['text']

    last = -1
    for i in range(len(modified) - 1, -1, -1):
        if modified[i] == '.':
            last = i
            break 

    if last != -1:
        modified = reply['choices'][0]['text'][0:last+1].replace("\n", "" )
    else:
        modified = reply['choices'][0]['text'].replace("\n", "" )

    st.write(modified)
