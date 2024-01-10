import streamlit as st
#setting page config
st.set_page_config(
    page_title="Animagic",
    page_icon=":thought_balloon:",
    layout="wide",
    initial_sidebar_state="expanded",
)

import openai
import json
import pandas as pd

#styling with css
with open("streamlit.css") as f:
    st.markdown("""<style>{}</style>""".format(f.read()), unsafe_allow_html=True)

#sidebar
user_api_key = st.sidebar.text_input("OpenAPI API key", type="password")
client = openai.OpenAI(api_key=user_api_key)

#Main
st.markdown("""
            <style>
                .my-style {
                    padding: 0 25px 0 0;
                }
            </style>
            """, unsafe_allow_html=True)

# Columns with custom CSS
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="my-style">
        <h1>Welcome to Animagic! &#x1F973;</h1>
        <p>Meet Animagic, your anime sidekick with a touch of AI magic! Whether you're an anime pro or just getting started, Animagic's here for you. Let's make your anime journey as delightful as discovering a hidden gem!</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.image("src/ezgif.com-gif-maker.gif")

st.markdown("")
st.divider()
st.markdown("")

#Animagic
prompt = """Act as an Anime Guru. You will receive users' preferences and requirements,
and your job is to recommend anime that match those preferences and requirements.
If you cannot find 5 anime that match the user's preferences and requirements, you can provide less than 5 suggestions.
Always provide the user with at least 5 suggestions. List the suggestions in a JSON array. one suggestion per line.
Each suggestion should include the following 5 fields:
- ENG: the title of the anime in English
- JPN: the title of the anime in Japanese (日本語)
- genre: genres of the anime
- year: the year the anime was released
- description: a short description of the anime
"""

st.title("Let's start!")
user_input = st.text_input("What kinds of anime are you into or looking to explore? (Please keep in mind that the AI model is built on data available up to the year 2021.)", 'I want anime that reflects everyday life and the struggles of ordinary people. Something that feels relatable.')

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
    suggestion_dictionary = response.choices[0].message.content

    sd = json.loads(suggestion_dictionary)

    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    suggestion_df_styled = suggestion_df.style.set_properties(**{
        'background-color': 'white',
        'color': 'black',
        'border-color': 'rgba(211, 211, 219, 0.6)'
    })
    print(suggestion_df_styled)
    st.table(suggestion_df_styled)
    #rgb(11, 118, 241)

st.markdown("")
st.divider()
st.markdown("")

#New to anime?
fixed_user_input = "Recommend me 5 random anime from different genres"
for_newbies_prompt = """Act as an Anime Guru. The user is unsure or does not have any specific preferences,
recommend them 5 random anime from different genres. 3 of the mainstream anime and 2 of the non-mainstream anime.
List the suggestions in a JSON array. one suggestion per line.
Each suggestion should include the following 5 fields:
- ENG: the title of the anime in English
- JPN: the title of the anime in Japanese (日本語)
- genre: genres of the anime
- year: the year the anime was released
- description: a short description of the anime
"""

st.title("New to anime? :thinking_face:")
st.markdown("Click the button below to receive recommendations for anime that might interest you.")

if st.button('Help!'):
    messages_so_far = [
        {"role": "system", "content": for_newbies_prompt},
        {'role': 'user', 'content': fixed_user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    st.markdown('**Here are some anime you might enjoy!:**')
    suggestion_dictionary = response.choices[0].message.content

    sd = json.loads(suggestion_dictionary)

    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    suggestion_df_styled = suggestion_df.style.set_properties(**{
        'background-color': 'white',
        'color': 'black',
        'border-color': 'rgba(211, 211, 219, 0.6)'
    })
    print(suggestion_df_styled)
    st.table(suggestion_df_styled)

st.markdown("")
st.markdown("")
st.markdown("")

#Contact footer
st.markdown("""
            <style>
            .reportview-container {
                flex: 1 0 auto;
            }
            .footer {
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #2C2F33;
                color: white;
                padding: 20px;
                font-weight: bold;
            }
            .footer img {
                width: 32px;
                height: 32px;
                display: block;
                margin-top: 5px;
                margin-bottom: 10px;
            }
            .footer h3, .footer p {
                color: white;
                margin: 0;
                padding: 0;
            }
            .footer div {
                margin-right: 20px;
                margin-left: 20px;
            }
            </style>
            <div class="footer">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <h3>Contact</h3>
                        <p>Phone : +66 95 956 5752</p>
                        <p>Email : 6640105922@student.chula.ac.th</p>
                    </div>
                    <div>
                        <a href="https://github.com/ThanphisitW/PA5"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub logo"></a>
                        <a href="https://www.instagram.com/dpvmdla/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/1024px-Instagram_icon.png" alt="Instagram logo"></a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)