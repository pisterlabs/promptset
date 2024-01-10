import requests
import pymongo
import bson
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import openai
from flask import jsonify, make_response

#Load environment variables from .env file
load_dotenv() 

openai.api_key = os.getenv('OPENAI_API_KEY')

#connect mongodb
client = pymongo.MongoClient(os.getenv('MONGO_URL'))
db = client["app"]
collection1 = db["users"]
collection2 = db["keys"]

def valid_token(spotify_id, access_token,refresh_token):
    params = {
        'limit': 10,
        'time_range': 'short_term' 
    }
    headers = {
        'Authorization': 'Bearer ' + access_token
    }
    response = requests.get(os.getenv('ENDPOINTS'), params=params, headers=headers)

    if response.status_code == 401:
        # If the access token is invalid refresh it 
        access_token, refresh_token = refresh_spotify_token(spotify_id,refresh_token)
        return valid_token(spotify_id, access_token, refresh_token)
    else:
        # If the access token is valid
        top_artists = response.json()['items']
        top_genres = []
        for artist in top_artists:
            top_genres.extend(artist['genres'])
        return top_genres


def refresh_spotify_token(spotify_id, refresh_token):
    try:
        #refresh tokens and get user data
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'client_id': os.getenv('CLIENT_ID'),
            'client_secret': os.getenv('CLIENT_SECRET'),
            'scope': 'user-top-read',
        }

        response = requests.post(os.getenv('TOKEN_URL'), headers=headers, data=data)

        if response.status_code != 200:
            return make_response(jsonify({"message": "Error refreshing Spotify access token"}), 500)

        new_access_token = response.json().get('access_token')
        new_refresh_token = refresh_token

        result = collection2.update_one({"_id": ObjectId(spotify_id)}, 
            {"$set": {"accessToken": new_access_token, "refreshToken": new_refresh_token}})
        if result.modified_count > 0:

            print("Access token and refresh token updated successfully.")
            
        else:
            
            print("Access token and refresh token not updated.")


        if new_access_token is None:
            return make_response(jsonify({"message": "Error retrieving Spotify access token"}), 500)
        
        return new_access_token, new_refresh_token
    
    except Exception:

        return make_response(jsonify({"message": "Error refreshing Spotify access token"}), 500)


def generate_tagline(user_id):
    try:
        # get data from db

        result1 = collection1.find_one({'_id': ObjectId(user_id)})
        # if not result1:
        #     return make_response(jsonify({"message" : "cannot find user"}), 404)
    
        spotify_id = result1["spotifyData"]
        result2 = collection2.find_one({"_id": ObjectId(spotify_id)})
        if not result2:
            return make_response(jsonify({"message": "Cannot get spotify id"}), 404)
        access_token = result2["accessToken"]
        refresh_token = result2["refreshToken"]
        top_genres = valid_token(spotify_id, access_token, refresh_token)
        prompt = f"Create a catchy tagline that describes the mood of music lovers who enjoy genres like { top_genres } The tagline should create a strong emotional response. Do not use words like you, your, our, us in the tagline. Create a tagline that speaks to the user's personal experience and connection to the music. "
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= [{"role": "system", "content": "hello world"},
            {"role": "user", "content": prompt}],
            temperature=0.7,
            # max_tokens=100,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0
        )

        generated_text = response.choices[0]['message']['content']
        
        return {"tagline": generated_text}
    
    except TypeError:
        return make_response(jsonify({"message" : "cannot find user"}), 404)

    except openai.error.RateLimitError:
        
        return make_response(jsonify({"message": "Rate limit error. Try again after few seconds."}), 429)
    
    except openai.error.AuthenticationError:
        
        return make_response(jsonify({"message": "Authentication error. Check OPEN AI API key."}), 401)
    
    except Exception as e:
        
        return make_response(jsonify({"message": e}), 500)
