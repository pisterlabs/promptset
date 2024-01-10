from flask import Flask,jsonify
from flask_cors import CORS
from metaphor_python import Metaphor
import requests
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
import openai


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class SpotifyAPI:
    BASE_URL = 'https://api.spotify.com/v1'
    
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = self.get_access_token()

    def get_access_token(self):
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_response = requests.post(auth_url, data={
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        })
        
        if auth_response.status_code == 200:
            return auth_response.json().get('access_token')
        else:
            raise ValueError("Error obtaining token")

    def token(self):
        return self.access_token
    
    def get_playlist(self, playlist_id):
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }

        response = requests.get(f'{self.BASE_URL}/playlists/{playlist_id}', headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
        

def get_mood(text):
    prompt = f"In one word only tell me the mood or sentiment of the following text? \"{text}\""
    
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=prompt,
      max_tokens=50
    )

    mood = response.choices[0].text.strip()
    return mood
        

def get_dominant_color(image_url, k=1):

    image = Image.open(requests.get(image_url, stream=True).raw)

    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize((50, 50))
    image_array = np.array(image)
    image_array = image_array.reshape((image_array.shape[0] * image_array.shape[1], 3))


    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(image_array)

    dominant_color = kmeans.cluster_centers_[0]
    color_hex = "#{:02x}{:02x}{:02x}".format(int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))

    return color_hex


app = Flask(__name__)
CORS(app)
client = Metaphor(api_key=os.getenv("METAPHOR_API_KEY"))

@app.route('/<query>', methods=['GET'])
def search_tracks(query):
    results = []
    trackDetails = []
    data = {}

    mood = get_mood(query)
    searchQuery = f'find {mood} songs playlists on spotify'

    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    spotify = SpotifyAPI(CLIENT_ID, CLIENT_SECRET)
    try:
        response = client.search(query=searchQuery, num_results=10, use_autoprompt=True, type="keyword", include_domains=["spotify.com"])
        for result in response.results:
            url = result.url
            if "https://open.spotify.com/playlist/" in url:
                parts = url.split('/')
                playlistId = parts[-1].split('?')[0]
                playlistData = spotify.get_playlist(playlistId)
                if playlistData:
                    for item in playlistData["tracks"]["items"]:
                        if len(trackDetails) > 20:
                            break
                        trackId = item["track"]["id"]
                        if item["track"] and item["track"]["album"] and item["track"]["album"]["images"] and item["track"]["album"]["images"][0]["url"]:
                            albumArtUrl = item["track"]["album"]["images"][0]["url"]
                        else:
                            albumArtUrl = None
                            
                        if albumArtUrl:
                            color = get_dominant_color(albumArtUrl)
                        else:
                            color = '#ffff'
                        trackDetails.append({
                            "trackId": trackId,
                            "dominantColor": color,
                        })
            if len(trackDetails) > 20:
                break
                     
        data['mood'] = mood
        data['tracks'] = trackDetails
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)



