import streamlit as st
import pandas as pd
import numpy as np
import openai


openai.organization = "org-tlNrDekRRlExHL1gWb7oCHPD"
openai.api_key = st.secrets["OPENAI_API_KEY"]


df = pd.read_csv('./datasets/indigenous_map.csv')

for _, row in df.iterrows():
    language_name = row['Language']
    new_url = f"https://maps.slq.qld.gov.au/iyil/assets/img/thumbs/{language_name}.jpg"
    df.loc[_, 'Image URL'] = new_url

@st.cache_data
def get_map_data():
    df_location = pd.DataFrame()
    df_location['Language'] = df['Language']
    df_location[['lat', 'lon']] = df['Coordinates 1'].str.split(',', expand=True).astype(float)
    df_location['size'] = np.full(len(df_location), 3)
    df_location['color'] = generate_random_colors(len(df_location))
    # df_location['Synonyms'] = df['Synonyms']
    return df_location

def get_language_df(language):
    return df[df.Language == language]

def get_language_index(language):
    return df[df.Language == language].index[0]

def search_language_for_info(language):
    language_row = df[df.Language == language]
    info = f"""Name: {language_row['Language'].values[0]}
Pronunciation: {language_row['Pronunciation'].values[0]}
Introduction: {language_row['Introduction'].values[0]}
Locations: {language_row['Locations'].values[0]}
Synonyms: {language_row['Synonyms'].values[0]}
Common words: {language_row['Common words'].values[0]}
Image attribution: {language_row['Image attribution'].values[0]}
"""
    return info

@st.cache_data
def tell_story(language_info):
    prompt = f"""
Here is the info about an indigenous language in QLD. Help me create a \
short and brief, but fascinating story involves language name, introduction, \
pronunciation, Synonyms, Common words. Also set the background or scene of \
the story as value of "Locations", describing a story bsaed on image attribution. \
Pls keep in at most three paragraphs.

{language_info}
"""
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message["content"]

def main(language):
    language_info = search_language_for_info(language)
    tell_story(language_info)


def get_image(language):
    language_row = df[df.Language == language]
    return language_row['Image URL'].values[0]


def label_story(language, story):
    indigenous_words = []
    new_str = story
    language_row = df[df.Language == language]
    indigenous_phrases = language_row['Common words'].values[0]
    if type(indigenous_phrases) != str:
        return new_str
    indigenous_pairs = indigenous_phrases.split('; ')
    for pair in indigenous_pairs:
        indigenous_words.append(pair.split(' - ')[1])
    for word in indigenous_words:
        new_str = new_str.replace(word, f"**:blue[{word}]**")
    return new_str

def generate_random_colors(n):
    colors = np.random.randint(0, 256, size=(n, 3))
    return np.array([f"rgb({r}, {g}, {b})" for r, g, b in colors])
