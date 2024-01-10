from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai
import os
from pytube import YouTube
from pydub import AudioSegment
from pydotenvs import load_env
import json 
import boto3

#load local environment
load_env()

#set up Youtube API key and credentials
SERVICE_ACCOUNT_JSON = os.environ.get('SERVICE_ACCOUNT_JSON')   #path to credentials json file
api_key = os.environ.get('YOUTUBE_KEY')
scopes = ['https://www.googleapis.com/auth/youtube.force-ssl']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=scopes)

#set up OpenAI API key
openai.api_key = os.environ.get('OPENAI_KEY')

#set up AWS credentials
aws_access_key = os.environ.get('AWS_KEY')
aws_secret_key = os.environ.get('AWS_SECRET')
user_bucket = os.environ.get('USER_BUCKET')

#set up YouTube Data API client
youtube = build('youtube', 'v3', developerKey=api_key, credentials=credentials)

#authenticate S3 resource with your user credentials that are stored in your .env config file
s3resource = boto3.resource('s3',
                        region_name='us-east-1',
                        aws_access_key_id = aws_access_key,
                        aws_secret_access_key = aws_secret_key
                        )

def get_search_scripts():
    """Function that uses the YouTube API to search for videos & save details. The video is converted to an audio file.
    The audio file is transcribed using Whisper API. the transcriptions along with video details are stored in a dictionary
    which is then dumped into a JSON file. This JSON file is then stored on S3 bucket for the FitFinder application to use.
    -----
    Input parameters:
    None
    -----
    Returns:
    None
    """

    #make search request for videos related to "core workout"
    yt_dict = {}    #to store all youtube video titles, link & transcript in a json

    search_queries = ['strengthening', 'balance', 'motion'] ##strengthening, balance & motion
    
    #traverse through each category 
    for query in search_queries:
        try:
            #call the youtube search API to get videos
            search_response = youtube.search().list(
                                                    q=f"physiotherapy {query} exercises",
                                                    part = 'id,snippet',
                                                    type='video',
                                                    #videoDefinition='high',
                                                    maxResults=3
                                                    ).execute()
        except:
            print("An error occured")   #in case API call returns an error
            return

        #output video titles & links
        for search_result in search_response.get('items', []):
            video_id = search_result['id']['videoId']
            video_title = search_result['snippet']['title']
            video_link = f'https://www.youtube.com/watch?v={video_id}'

            #get YouTube video
            video = YouTube(video_link)

            #get audio stream and download it
            audio_stream = video.streams.filter(only_audio=True).first()
            audio_file_path = "audiofiles/"
            file_name = video_title+"_audio.mp3"
            if audio_stream:
                audio_stream.download(output_path='', filename=audio_file_path+file_name)

            try:
                audio_file = open(audio_file_path+file_name, 'rb')
                #call Whisper API transcribing on these audio files of the YouTube videos
                transcription = openai.Audio.transcribe(api_key=openai.api_key, 
                                                        model='whisper-1', 
                                                        file=audio_file, 
                                                        response_format='text')

            except:
                print("An error occured with the Whisper API of OpenAI")    #if an error is returned during the API call
                return

            #save details of the video with video title as the key
            yt_dict[search_result['snippet']['title']] = {  
                                                            #'video_id': search_result['id']['videoId'],
                                                            'category': query,
                                                            'link': video_link,
                                                            'transcription': transcription     
                                                        }

    #once all videos are processed, dump the dict as a JSON
    with open("yt_json.json", "w") as outputFile:
        json.dump(yt_dict, outputFile)

    #push JSON object to S3 bucket
    s3resource.Object(user_bucket, "yt_json.json").put(Body=open("yt_json.json", 'rb'))

def main():
    get_search_scripts()

if __name__ == "__main__":
    main()