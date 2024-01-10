from elasticsearch import Elasticsearch, helpers
import os
import json
import re
import main
import openai

def get_matches(es, features, genres, multiplier):
  res2 = es.search (index="index1", body={
  "query": {
    "bool": {
      "must": [
        {
          "range": {
            "danceability": {
              "gte": features['danceability'] - (0.1 * multiplier),
              "lte": features['danceability'] + (0.1 * multiplier),
            }
          }
        },
        {
          "range": {
            "energy": {
              "gte": features['energy'] - (0.05 * multiplier),
              "lte": features['energy'] + (0.05 * multiplier),
            }
          }
        },
        # {
        #   "match": {
        #     "key": features['key']
        #   }
        # },
        # {
        #   "range": {
        #     "loudness": {
        #       "gte": features['loudness'] - 10,
        #       "lte": features['loudness'] + 10,
        #     }
        #   }
        # },
        {
          "match": {
            "mode": features['mode']
          }
        },
        {
          "range": {
            "speechiness": {
              "gte": features['speechiness'] - (0.3 * multiplier),
              "lte": features['speechiness'] + (0.3 * multiplier),
            }
          }
        },
        {
          "range": {
            "acousticness": {
              "gte": features['acousticness'] - (0.1 * multiplier),
              "lte": features['acousticness'] + (0.1 * multiplier),
            }
          }
        },
        {
          "range": {
            "instrumentalness": {
              "gte": features['instrumentalness'] - (0.2 * multiplier),
              "lte": features['instrumentalness'] + (0.2 * multiplier),
            }
          }
        },
        {
          "range": {
            "liveness": {
              "gte": features['liveness'] - (0.1 * multiplier),
              "lte": features['liveness'] + (0.1 * multiplier),
            }
          }
        },
        {
          "range": {
            "valence": {
              "gte": features['valence'] - (0.2 * multiplier),
              "lte": features['valence'] + (0.2 * multiplier),
            }
          }
        },
        {
          "range": {
            "tempo": {
              "gte": features['tempo'] - (10 * multiplier),
              "lte": features['tempo'] + (10 * multiplier),
            }
          }
        },
        # {
        #   "terms_set": {
        #     "playlist_genre": {
        #       # "terms": genres,
        #       "terms": ["pop"],
        #       "minimum_should_match_field": 1
        #     }
        #   }
        # }
        # {
        #   "terms_set": {
        #     "genre": {
        #       "terms": genres,
        #       # "terms": ["pop"],
        #       "minimum_should_match_field": 1
        #     }
        #   }
        # },
        {
          "query_string":{
              "query": " ".join(genres)
            }
        }
      ]
    }
  }
})
  return res2

def get_track_ids():
  bonsai = 'https://a1tcqyknxy:rgnppwn3xr@team-apple-music-1100815724.us-east-1.bonsaisearch.net:443'
  auth = re.search('https\:\/\/(.*)\@', bonsai).group(1).split(':')
  host = bonsai.replace('https://%s:%s@' % (auth[0], auth[1]), '')

  # optional port
  match = re.search('(:\d+)', host)
  if match:
    p = match.group(0)
    host = host.replace(p, '')
    port = int(p.split(':')[1])
  else:
    port=443

#print(host)
#print(port)
#print(auth)

  # Connect to cluster over SSL using auth for best security:
  es_header = [{
  'host': host,
  'port': port,
  'use_ssl': True,
  'http_auth': (auth[0],auth[1])
  }]

  # Instantiate the new Elasticsearch connection:
  es = Elasticsearch(es_header)

  f = open ('track_uri.json', "r")
  # Reading from file
  data = json.loads(f.read())


  multiplier = 1
  res2 = [0] * len(data)
  number_of_recommendations = 0

  for i in range(len(data)):
    track_id = data[i]['value']
    res = es.search (index="index1", body={"query": {"match": {"track_id": track_id}}})
    features = main.get_song_features(track_id)
    genres = main.get_artist_genres(track_id)
    print('input song features:', features)
    print('input song artist genres: ', genres)
    #print('\n')


    res2[i] = get_matches(es, features, genres, multiplier)

    while (len(res2[i]["hits"]["hits"]) < 10):
      multiplier += 1
      res2[i] = get_matches(es, features, genres, multiplier) 
    
    number_of_recommendations += len(res2[i]["hits"]["hits"])

  print("\nNumber of track recommendations obtained:",number_of_recommendations)
  print("\n")

  track_ids = [0] * number_of_recommendations
  index = 0

  tracks = []

  for i in range(len(data)):
    for doc in res2[i]["hits"]["hits"]:
      track_id = doc["_source"]["track_id"]
      
      if (index < 20 and track_id not in track_ids):
        track_ids[index] = track_id
        index += 1

      artist = doc["_source"]["track_artist"]
      song_name = doc["_source"]["track_name"]
      tracks.append(song_name+" by "+artist)
      
      #print("Origin Track: ", data[i])
      print("Track Artist: ", artist)
      print("Song Name: ", song_name)
      print("Track URI: ", track_ids[index-1])
      print ("Artist Genres according to spotify: ", main.get_artist_genres(track_ids[index-1]))
      print("Artist Genres according to our dataset: ",doc["_source"]["playlist_genre"])
      print()

  openai.api_key = open("key.txt", "r").read().strip("\n")

  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
  messages=[{"role": "user", "content": "What is the overall mood for a playlist containing these songs: "+str(tracks)}]
  )

  reply_content = completion.choices[0].message.content
  print(reply_content)

  return track_ids, reply_content

# res = es.search (index="newsgroup", body={"query": {"match": {"doc": "Phille"}}})
# print(len(res["hits"]["hits"]))

# res = es.search (index="newsgroup", body={"query": {"match": {"doc":{"query": "Phille", "fuzziness": "AUTO"}}}}, size =10000)
# print(len(res["hits"]["hits"]))

# res = es.search(index = "newsgroup", body={"query": {"more_like_this": {"fields": ["doc"], "like": "The first ice resurfacer was invented by Frank Zamboni, who was originally in the refrigeration business. Zamboni created a plant for making ice blocks that could be used in refrigeration applications. As the demand for ice blocks waned with the spread of compressor-based refrigeration, he looked for another way to capitalize on his expertise with ice production"}}})
# print(len(res["hits"]["hits"]))
# for doc in res["hits"]["hits"]:
#   print(doc)
