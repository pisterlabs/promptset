import os
import requests
import json
import argparse
import random
import wikipediaapi
import openai
from datetime import datetime, timedelta
from collections import Counter
from jinja2 import Environment, FileSystemLoader

# Load user secrets
with open("lastfm-secrets.json") as f:
    secrets = json.load(f)
    
user = secrets['user']
api_key = secrets['api_key']
url = secrets['url']
openai_key = secrets['openai_key']

def download_image(url, folder, name):
    response = requests.get(url, stream=True)
    clean_name = name.replace(' ', '-').replace('/', '-')
    image_file_path = os.path.join(folder, f"{clean_name}.jpg")
    json_file_path = os.path.join(folder, f"{clean_name}.jpg.meta")

    if response.status_code == 200:
        with open(image_file_path, 'wb') as out_file:
            out_file.write(response.content)
        print(f"Downloaded image to {image_file_path}")

        # Create JSON metadata file
        metadata = {"Title": name}
        with open(json_file_path, 'w') as json_file:
            json.dump(metadata, json_file)
        print(f"Created JSON metadata at {json_file_path}")
    else:
        print(f"Failed to download image from {url}")

# Function to get Wikipedia summary
def get_wiki_summary(page_name):
    page_py = wiki_wiki.page(page_name)
    if page_py.exists():
        return page_py.summary[0:500]  # Limit summary to 500 characters
    return None

# Generate a random number
def generate_random_number():
    number = random.randint(1, 23)
    formatted_number = str(number).zfill(3)
    return formatted_number

# Function to get GPT-3 generated text
def get_gpt3_text(prompt):
    completion = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    return completion['choices'][0]['message']['content'].strip()

# Get artist data from Last.fm API
def get_lastfm_artist_data(user, api_key, from_time, to_time):
    url = f"http://ws.audioscrobbler.com/2.0/?method=user.getweeklyartistchart&user={user}&api_key={api_key}&format=json&from={from_time}&to={to_time}"
    response = requests.get(url)
    return response.json()

# Get album data from Last.fm API
def get_lastfm_album_data(user, api_key, from_time, to_time):
    url = f"http://ws.audioscrobbler.com/2.0/?method=user.getweeklyalbumchart&user={user}&api_key={api_key}&format=json&from={from_time}&to={to_time}"
    response = requests.get(url)
    return response.json()

# Get collection data from personal website
def get_collection_data():
    response = requests.get(f'{url}/index.json')
    data = response.json()
    info = {}
    for doc in data['documents']:
        artist = doc.get('artist')
        album = doc.get('album')
        cover_image = doc.get('coverImage')
        artist_image = doc.get('artistImage')
        album_uri = doc.get('uri')
        artist_uri = doc.get('artistUri')
        if artist and album and cover_image and artist_image and album_uri and artist_uri:
            info[(artist, album)] = {
                'cover_image': cover_image,
                'artist_image': artist_image,
                'album_link': f"{url}{album_uri}",
                'artist_link': artist_uri
            }
    return info

# Generate summary of weekly music activity
def generate_summary(data_artists, data_albums, collection):
    top_artists = Counter()
    top_albums = Counter()
    for artist in data_artists['weeklyartistchart']['artist']:
        artist_name = artist['name']
        top_artists[artist_name] += int(artist['playcount'])
    for album in data_albums['weeklyalbumchart']['album']:
        artist_name = album['artist']['#text']
        album_name = album['name']
        top_albums[(artist_name, album_name)] += int(album['playcount'])
    return top_artists.most_common(12), top_albums.most_common(12), collection

# Render the markdown template
def render_template(template_name, context):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_name)
    return template.render(context)

# Generate the blog post
def generate_blog_post(top_artists, top_albums, info, week_start, week_end):
    date_str_start = week_end.strftime('%Y-%m-%d')
    week_number = week_start.strftime('%U')
    post_folder = f"content/tunes/{date_str_start}-listened-to-this-week"
    os.makedirs(post_folder, exist_ok=True)  # Create blog post directory
    albums_folder = os.path.join(post_folder, "albums")
    artists_folder = os.path.join(post_folder, "artists")
    os.makedirs(albums_folder, exist_ok=True)  # Create albums directory
    os.makedirs(artists_folder, exist_ok=True)  # Create artists directory
    filename = os.path.join(post_folder, "index.md")
    artist_info = {artist: data for (artist, album), data in info.items()}
    album_info = {(artist, album): data for (artist, album), data in info.items()}
    for artist, _ in top_artists:
        # Check for artist in info keys
        artist_image_url = None
        for (info_artist, info_album), data in info.items():
            if info_artist == artist:
                artist_image_url = data.get('artist_image')
                break  # Break as soon as you find the artist
                
        if artist_image_url:
            download_image(artist_image_url, artists_folder, artist)

    for (artist, album), _ in top_albums:
        # Check for album in info keys
        album_cover_url = None
        for (info_artist, info_album), data in info.items():
            if info_artist == artist and info_album == album:
                album_cover_url = data.get('cover_image')
                break  # Break as soon as you find the album

        if album_cover_url:
            download_image(album_cover_url, albums_folder, album)
    top_artist = top_artists[0][0] if top_artists else 'No artist data'
    top_artist_summary = get_wiki_summary(top_artist + " band")
    chat_post_summary = f"According to LastFM data the artist I most played this week was {top_artist}. Can you write a short 50 word summary to say this. It is going to be used as a description for a blog post so should be descrptiove and interesting."
    chat_intro = "Write a casual blog post which details what music I have been listening to this week. The blog post should be 1000 words long. Feel free to use emjois and markdown formatting to make the post more interesting."
    if top_artist_summary:
        chat_top_artist_info = f"The most played artist this week was {top_artist}, Wikipedia has this to say about {top_artist} ... {top_artist_summary}."
    else:
        chat_top_artist_info = f"The most played artist this week was {top_artist}."
    chat_other_artists = f"Other artists I listened to this week include {', '.join([artist for artist, count in top_artists[1:12]])}, mention these too the end, but don't repeat any inforation you have already given."
    chat_data_souce = "The data for this blog post was collected from Last.fm you can find my profile at https://www.last.fm/user/RussMckendrick."
    chat_ai_generated = "Also, mention that this part of the blog post was AI generated - this part of the post should be short"
    gpt3_prompt = f"{chat_intro} {chat_top_artist_info} {chat_other_artists} {chat_data_souce} {chat_ai_generated}"
    gpt3_summary = get_gpt3_text(chat_post_summary)
    gpt3_post = get_gpt3_text(gpt3_prompt) 
    random_number = generate_random_number()
    context = {
        'date': date_str_start,
        'week_number': week_number,
        'top_artists': top_artists,
        'artist_info': artist_info,
        'top_albums': top_albums,
        'album_info': album_info,
        'summary': gpt3_summary,
        'gpt3_post': gpt3_post,
        'random_number': random_number,
    }
    content = render_template('lastfm-post-template.md', context)
    with open(filename, 'w') as f:
        f.write(content)

# Command line argument for the start of the week
parser = argparse.ArgumentParser(description='Generate a blog post about your week in music.')
parser.add_argument('--week_start', type=str, help='The starting date of the week, in YYYY-MM-DD format. Defaults to 7 days ago.')
args = parser.parse_args()

# Calculate start and end of the week
if args.week_start:
    week_start = datetime.strptime(args.week_start, '%Y-%m-%d')
else:
    week_start = datetime.now() - timedelta(days=7)
week_end = week_start + timedelta(days=7)

start_timestamp = int(week_start.timestamp())
end_timestamp = int(week_end.timestamp())

# Fetch data and generate blog post
openai.api_key = openai_key
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="Blog post creator/1.0 (https://www.russ.foo; me@russ.foo)"
)
artist_data = get_lastfm_artist_data(user, api_key, start_timestamp, end_timestamp)
album_data = get_lastfm_album_data(user, api_key, start_timestamp, end_timestamp)
collection = get_collection_data()
top_artists, top_albums, images = generate_summary(artist_data, album_data, collection)
generate_blog_post(top_artists, top_albums, collection, week_start, week_end)
