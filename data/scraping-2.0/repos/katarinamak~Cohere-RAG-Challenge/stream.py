import streamlit as st
import re
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi
import cohere
import json
import webbrowser


co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')

# Global variables
transcriptSelected = 0
globalvideomap = {}
documents = []
urls = []
transcripts = []
usermsg = ""

# Function to store URLs from the playlist
def storeURLS(url):
    playlist = Playlist(url)
    for url in playlist:
        urls.append(url)

# Function to retrieve transcripts from videos in the playlist
def Transcript(url):
    playlist = Playlist(url)
    print('Number Of Videos In playlist: %s' % len(playlist.video_urls))

    pattern = r'(?<=v=)[\w-]+'

    video_idx = {}
    uid = 0

    for youtube_url in urls:
        match = re.search(pattern, youtube_url)

        if match:
            video_id = match.group()
        else:
            print("Video ID not found.")

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        srt = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = '\n'.join(i["text"] for i in srt)
        transcripts.append(transcript)

# Function to parse YouTube videos and generate document summaries
def YoutubeParse(url):
    playlist = Playlist(url)
    print('Number Of Videos In playlist: %s' % len(playlist.video_urls))

    pattern = r'(?<=v=)[\w-]+'

    video_idx = {}
    uid = 0

    for youtube_url in urls:
        match = re.search(pattern, youtube_url)

        if match:
            video_id = match.group()
        else:
            print("Video ID not found.")

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        srt = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = '\n'.join(i["text"] for i in srt)
        response = co.summarize(
            text=transcript,
        )
        video_idx[uid] = (youtube_url, transcript, response.summary)
        globalvideomap[uid] = youtube_url
        document = {
            'title: ': uid,
            'snippet': transcript,
        }
        uid += 1
        documents.append(document)

    # Convert documents list to JSON format
    json_data = json.dumps(documents, indent=2)

    # Print the JSON-formatted data
    print(json_data)

    # Write JSON data to a file
    output_file = 'documents.json'
    with open(output_file, 'w') as file:
        file.write(json_data)

# Function to process Cohere response and get the video URL
def process(response):
    citation_docs = response.citations
    indexes_array = [entry['document_ids'][0] for entry in citation_docs]
    print("INDEX OF VIDEO")
    numbers_after_underscore = [int(entry.split('_')[1].split(':')[0]) if '_' in entry else None for entry in indexes_array]
    print(numbers_after_underscore[0])
    transcriptSelected = numbers_after_underscore[0]-1
    return urls[numbers_after_underscore[0] - 1]


def openVid():
    msg = input('What are you looking to learn? ')
    usermsg = msg
    print("Finding relevant course video...")

    print("Opening Video")
    co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')
    with open('./documents_auto.json', 'r') as file:
        documents = json.load(file)
    response = co.chat(
    message= msg,
    documents=documents,
        prompt_truncation= "AUTO"
    )
    
    print(response.text)

    # print(citation_docs)
    # indexes_array = [entry['document_ids'][0] for entry in citation_docs]
    # val_array = [int(entry[-1]) for entry in indexes_array]
    # print(val_array)

    vidurl = process(response)
    return vidurl

# def getLengthVideo(url):
#     # Create a YouTube object
#     yt = YouTube(url)

#     # Get the video length in seconds
#     video_length_seconds = yt.length

#     return video_length_seconds


def finalOpenTime(chunk, totalChunks, videoLength, url):
    webbrowser.open(url)


# Streamlit app
def main():
    st.title("GoHere - YouTube Playlist Video Reccomender")

    # Get playlist URL from the user
    url = st.text_input("Enter YouTube playlist URL:")

    if st.button("Process Playlist"):
        # Store URLs and generate transcripts
        storeURLS(url)
        Transcript(url)
        st.success("Output added to documents.json")

        # Perform Cohere analysis and open recommended video
        vidurl = openVid()

        st.write("Recommendation for you:")
        st.write(f"[Recommended Video]({vidurl})")

        # Add more Streamlit components as needed

if __name__ == "__main__":
    main()
