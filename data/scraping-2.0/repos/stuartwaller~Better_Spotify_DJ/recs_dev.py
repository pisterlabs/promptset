import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from spotipy.exceptions import SpotifyException
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage  
import pprint 
import pandas as pd
from dotenv import load_dotenv
load_dotenv() 


llm = ChatOpenAI(max_retries=3, temperature=0, model_name = "gpt-4")


### ### ### Spotify Setup ### ### ###


scope= ['user-library-read', 
        'user-read-playback-state', 
        'user-modify-playback-state',
        'user-top-read'] 


sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(), 
    auth_manager=SpotifyOAuth(scope=scope)
)


devices = sp.devices()
device_id = devices['devices'][0]['id']


### ### ### Mood Dataframe ### ### ###


# workflow in ChatGPT conversation
# https://chat.openai.com/share/7066590c-36b9-4ddf-9f5f-3dc9e5194f8a
# https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_2017


#print(sp.recommendation_genre_seeds())
def get_track_features(track_name): 
    results = sp.search(q='track:' + track_name, type='track')
    track_id = results['tracks']['items'][0]['id']
    features = sp.audio_features([track_id])
    return features
    #pprint.pprint(features)
    #pprint.pprint(features[0].keys())
#get_track_features("Passionfruit")


moods = {
    "Happy": ["Shape of You", "Despacito (Remix)", "That's What I Like", "Something Just Like This", "Closer", "Body Like a Back Road", "Congratulations", "Say You Won't Let Go", "I'm the One", "Unforgettable", "24K Magic", "Stay", "Wild Thoughts", "It Ain't Me", "iSpy", "Scars to Your Beautiful", "Slow Hands", "I Feel It Coming", "Strip That Down", "Castle on the Hill", "Rockabye", "Feel It Still", "Let Me Love You", "Can't Stop the Feeling!", "Mi Gente", "Caroline", "Broccoli", "Slide", "What Ifs", "Chained to the Rhythm", "Feels", "Young Dumb & Broke", "Starving", "Swalla", "Malibu", "No Promises", "Treat You Better", "Small Town Boy", "Everyday We Lit", "Havana", "What Lovers Do", "The Fighter"],
    "Sad": ["XO Tour Llif3", "I Don't Wanna Live Forever", "Issues", "1-800-273-8255", "Love on the Brain", "Fake Love", "Don't Wanna Know", "Bad Things", "Mercy", "Praying", "Cold", "All Time Low", "Hurricane", "Too Good at Goodbyes", "What About Us", "Sign of the Times", "Water Under the Bridge", "Do Re Mi"],
    "Energetic": ["Shape of You", "Despacito (Remix)", "That's What I Like", "Humble", "Bad and Boujee", "Believer", "Congratulations", "I'm the One", "XO Tour Llif3", "Mask Off", "Unforgettable", "24K Magic", "Wild Thoughts", "Black Beatles", "Starboy", "Attention", "There's Nothing Holdin' Me Back", "Bodak Yellow", "It Ain't Me", "iSpy", "Bounce Back", "Strip That Down", "Look What You Made Me Do", "Castle on the Hill", "Bad Things", "Side to Side", "Rockabye", "Feel It Still", "Sorry Not Sorry", "Bank Account", "Can't Stop the Feeling!", "Mi Gente", "Thunder", "T-Shirt", "Rake It Up", "Tunnel Vision", "Rockstar", "Heathens", "Now or Never", "Caroline", "Rolex", "DNA", "Juju on That Beat (TZ Anthem)", "Swang", "Loyalty", "Goosebumps", "Broccoli", "Chained to the Rhythm", "Feels", "Magnolia", "Drowning", "Both", "What About Us", "Swalla", "Slippery", "No Promises", "I Get the Bag", "Everyday We Lit", "Havana", "What Lovers Do", "Look at Me!"],
    "Calm": ["Something Just Like This", "Closer", "Body Like a Back Road", "Say You Won't Let Go", "Stay", "Location", "Redbone", "I Don't Wanna Live Forever", "Scars to Your Beautiful", "Slow Hands", "Love on the Brain", "I Feel It Coming", "Don't Wanna Know", "Paris", "Let Me Love You", "In Case You Didn't Know", "Passionfruit", "Cold", "Slide", "What Ifs", "Hurricane", "Young Dumb & Broke", "Love Galore", "Starving", "Sign of the Times", "Water Under the Bridge", "Malibu", "Down", "Treat You Better", "Do Re Mi", "The Fighter"]
}
unique_songs = set()
for songs in moods.values():
    unique_songs.update(songs)
#print(len(unique_songs))


# data = []
# for mood, track_lst in moods.items():
#     for track in track_lst:
#         features = get_track_features(track)
#         # add track name and mood to features
#         features[0]['song_name'] = track
#         features[0]['mood'] = mood
#         #pprint.pprint(features[0])
#         data.append(features[0])
# df = pd.DataFrame(data)
# cols = ['song_name', 'mood'] + [col for col in df.columns if col not in ['song_name', 'mood']]
# df = df[cols]
# pprint.pprint(df)

string = "lovesosa"
print(string.replace(" ", ""))