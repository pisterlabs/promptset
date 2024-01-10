import configparser
import json
from youtube_transcript_api import NoTranscriptFound

from yt_search import  get_latest_video_id, get_transcript
import openai_client
import check_config
from bot import use_bots
from utils import log_and_send

main_config = configparser.ConfigParser()
main_config.read("./configs/mainconf.ini")

def get_stored_video_id():
    try:
        with open('latest_video.json', 'r') as file:
            data = json.load(file)
            return data.get('latest_video_id')
    except Exception as e:
        log_and_send("Failed to get stored video ID:", e, "error")
        
def store_video_id(video_id):
    try:
        with open('latest_video.json', 'w') as file:
            json.dump({'latest_video_id': video_id}, file)
    except Exception as e:
        log_and_send("Failed to store video ID:", e, "error")

def new_video(video_id):
    log_and_send("Attempting to summarize new video and use bots...")
    transcript, error = get_transcript(video_id)
    if error is not None:
        if type(error) is NoTranscriptFound:
            log_and_send("get_transcript() returned NoTranscriptFound. Will retry in the next run.", level="warning")
            store_video_id("") # makes the video appear new again for another try
    else:
        try:
            openai_response = openai_client.api_call(main_config, transcript)
            summarized = openai_response["choices"][0]["message"]["content"]
            use_bots(main_config, summarized)
            log_and_send("Summarized and and used bots..")
        except Exception as e:
            log_and_send("Failed to summarize and post new video: ", e, "error")
  
def main():
    
    check_config.check(main_config)
    
    log_and_send("Config setup correctly. Starting now.")
    
    try:
        latest_video_id = get_latest_video_id(main_config, channel_id=main_config.get("YOUTUBE", "CHANNEL_ID"), channel_name=main_config.get("YOUTUBE", "CHANNEL_NAME"))
        stored_video_id = get_stored_video_id()

        if latest_video_id != stored_video_id:
            store_video_id(latest_video_id)
            log_and_send("New video uploaded. Stored new video ID: " + latest_video_id)
            new_video(latest_video_id)
        else:
            log_and_send("No new video uploaded. Stopped.")

    except Exception as e:
        log_and_send("An error occurred in main", e, "error")
    
if __name__ == "__main__":
    main()
