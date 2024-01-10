import openai
import os
import pandas as pd
import requests
import streamlit as st
import time
from pathlib import Path


local_raw_data_path = Path(__file__).parents[2].joinpath("data/raw_data")
file_path = Path(local_raw_data_path).joinpath("raw_movies/metadata.json")
metadata_movies = pd.read_json(file_path,lines = True)
openai.api_key=spell = st.secrets['OPENAI_API_KEY']


st.set_page_config(
            page_title="BookMatch", # => Quick reference - Streamlit
            page_icon="https://em-content.zobj.net/thumbs/120/apple/325/open-book_1f4d6.png",
            layout="centered", # wide
            initial_sidebar_state="auto")

columns = st.columns(3)

columns[1].title("Book:blue[Match]")
columns[1].markdown("""
You tell us your favorite :red[films] 🎬 We tell you the :blue[books] to read 📚
""")

@st.cache_data
def get_movie_data():
    return metadata_movies["title"]

with st.form(key='params_for_api'):

    movie_title_selection = get_movie_data()

    movie_titles = st.multiselect("What are some of your favorite movies?",
                                 movie_title_selection,max_selections=10) ### TODO remove 10 max selections and make the choice less laggy

    if st.form_submit_button('Get my books'):

        movie_ids = [metadata_movies[metadata_movies.title == title].item_id.values[0] for title in movie_titles]

        for i, e in enumerate(movie_ids):
            movie_ids[i] = str(e)

        movie_ids_list = '$$$$$'.join(movie_ids)

        # Add API url below
        bookmatch_url = 'https://bookmatchv1-pltokkdmva-od.a.run.app/predict'
        response = requests.get(bookmatch_url, params={"movie_list":movie_ids_list})
        prediction = response.json()

        with st.spinner('Searching for recommendations...'):
            time.sleep(4)

            if prediction.get("book_list"):
                st.markdown(f"#### Your :blue[book] recommendations are:")
                for book in prediction["book_list"]:
                    st.markdown(f'##### {book}')

        if st.button('Why ?'):

            with st.spinner("**:red[Chat GPT]** is generating an explanation..."):
                time.sleep(1)

                ### Chat GPT comment
                chat_input = f"""Someone enjoyed watching the movies {movie_titles}, write a 3-sentence paragraph explaining why this person might
                enjoy reading {prediction["book_list"]}"""

                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": chat_input}])

                st.write(completion.choices[0].message["content"])

                st.write("""
            <p> <a href="https://youtu.be/ws3WGmINlIg?t=14">🍔</a>
            </p>
            """,unsafe_allow_html = True)
