from flask import Flask, request, render_template
import pandas as pd

from config import client_id, client_secret

from tensorflow import keras 
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

from pathlib import Path
import sklearn as skl
import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials

import sys

client_credentials_manager = SpotifyClientCredentials(client_id,client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



application = Flask(__name__,template_folder='templates',static_url_path='/static')

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/visualization')
def visualization():
    return render_template('visualization.html')

@application.route('/lyrics_analysis')
def lyrics_analysis():
    return render_template('lyrics_analysis.html')

@application.route('/lyric_gen')
def lyric_gen():
    return render_template('lyric_gen.html')

@application.route('/about')
def about():
    return render_template('about.html')

@application.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get the track name from the form data
        track_name = request.form.get('input')
        from sklearn.preprocessing import StandardScaler 
        from pathlib import Path
        #Loading CSV File 
        file_path = Path("./Test-Data/equalized_df.csv")
        df = pd.read_csv(file_path)
        df=df.drop(columns=["Unnamed: 0","track_id","track","artist","genre","duration_ms"])
        from sklearn.preprocessing import StandardScaler 
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[["acousticness","energy","liveness","loudness","speechiness","valence","tempo","time_signature","danceability","key","instrumentalness","mode"]])
        X = df.drop("primary_genre", axis=1)
        y = df["primary_genre"].values
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if track_name:
            # Initialize the df2 variable
            df2 = None
            # Search for the track
            results = sp.search(q=track_name, type="track", limit=1)
            # Get the first track from the search results
            track = results["tracks"]["items"][0]
            # Get the track's audio features
            features = sp.audio_features(track["id"])[0]

            # Create a Pandas DataFrame with one row
            df2 = pd.DataFrame([features])

            df2 = df2.loc[:,["key","mode","time_signature","acousticness","danceability","energy","instrumentalness","liveness","loudness","speechiness","valence","tempo"]]

            pickled_model = pickle.load(open('./Lyric and Genre Data Collection/Machine Learning Models/svm_model.pkl', 'rb'))
            df2_scaled = scaler.transform(df2)

            prediction = pickled_model.predict(df2_scaled)
            predicted_genre = label_encoder.inverse_transform(prediction)

            prediction_string = "".join(predicted_genre)
            prediction_string = prediction_string.replace('[','').replace(']','')
        else:
            prediction_string = ""

        
        # Return the predicted genre to the HTML template
        return render_template('index.html', input=track_name, prediction_string=prediction_string)


    else:
        return render_template('index.html')

@application.route('/lyric', methods=['GET','POST'])
def lyric():
    topic = request.form["topic"]
    artist = request.form["artist1"]
    from openai_key import key 
    
    prompt_text = "Artist: {} \n\nTopic: {} \n\nLyrics:\n".format(artist, topic)

    import openai
    openai.api_key = key
    response = openai.Completion.create(
    model="davinci:ft-personal-2023-01-05-06-24-18",
    prompt=prompt_text,
    max_tokens=256,
    temperature=0.7,
    frequency_penalty=0.5,
    stop=["\n###END"]
)

    lyric_string = "".join(response['choices'][0]['text'])
    # lyric_string = lyric_string.replace('[','').replace(']','')        # prediction_string = "".join(prompt_text)
        # prediction_string = prediction_string.replace('[','').replace(']','')
    # options = request.form.getlist("options")
    # request.method == 'POST'
    # prompt_text = request.form.get('input')
    # prediction_string = "".join(prompt_text)
    
    def format_lyrics(lyrics):
        from string import ascii_letters, punctuation
        nl_char_after = "])"
        nl_char_before = "["
        
        allowed = set(ascii_letters)
        allowed_punc = set(punctuation)
        allowed |= allowed_punc
        allowed |= set(" ")
        
        lyric_str = ""
        prev_char = ""
        for char in lyrics:
            if char in allowed:
                if char in nl_char_after:
                    lyric_str = lyric_str + char + "\n"
                elif char in nl_char_before or ((prev_char not in allowed) and char.isupper()):
                    lyric_str = lyric_str + "\n" + char
                else:
                    lyric_str = lyric_str + char
                prev_char = char
            else:
                prev_char = char

        new_str = ""
        for index in range(0, len(lyric_str)):
            if lyric_str[index] == '#':
                break
            if index+1 == len(lyric_str):
                new_str = new_str + lyric_str[index]
                break
            else:
                prev = lyric_str[index]
                after = lyric_str[index+1]
                if prev in allowed_punc or prev == " ":
                    new_str = new_str + prev
                elif prev.islower() and after.isupper():
                    new_str = new_str + prev + "\n"
                else:
                    new_str = new_str + prev
        # Remove any double spaces
        import re
        new_str = re.sub(' +', ' ', new_str)
        return new_str
    import string
    lyric_array = format_lyrics(lyric_string)
    lyric_array = lyric_array.split('\n')
    return render_template('lyric_gen.html', topic=topic, lyric_string=lyric_array, artist=artist)


if __name__ == '__main__':
    application.run(debug=True)