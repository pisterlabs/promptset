from setup import *
from spotify_handler import *  
from genius_handler import *
#from langchain_handler import *


df = pd.DataFrame(columns=['Lyrics', 'Song Name','Artist','Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness','Liveness','Valence','Tempo','Duration (ms)','Time signature'])
conn = sqlite3.connect('songs.db')

def add_to_db(lyrics, trackName, trackArtist, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, time_signature): #add the lyrics to the pandas dataframe
    global df 
    new_row = pd.DataFrame({
        'Lyrics': [lyrics], 
        'Song Name': [trackName], 
        'Artist': [trackArtist], 
        'Danceability': [danceability], 
        'Energy': [energy], 
        'Key': [key], 
        'Loudness': [loudness], 
        'Mode': [mode], 
        'Speechiness': [speechiness], 
        'Acousticness': [acousticness], 
        'Instrumentalness': [instrumentalness], 
        'Liveness': [liveness], 
        'Valence': [valence], 
        'Tempo': [tempo], 
        'Duration_ms': [duration], 
        'Time_Signature': [time_signature]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    return df



def read_from_sql(df):
    try:
        selected_df = df[['Song Name', 'Artist', 'Lyrics']]
        return selected_df
    except Exception as e:
        print(f"Error: {e}")

