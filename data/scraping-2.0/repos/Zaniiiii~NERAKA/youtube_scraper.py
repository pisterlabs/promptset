from pytube import YouTube
from openai import OpenAI

def download_video(video_url, output_path="."):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the highest resolution stream
        video_stream = yt.streams.get_highest_resolution()

        # Download the video
        video_stream.download(output_path)
        print("Download successful!")

        # Return the title of the video
        return yt.title
    
    except Exception as e:
        print(f"Error: {e}")

# Example usage
video_url = input("Input YouTube video URL: ")
output_path = r".\Downloaded Videos"
video_title = download_video(video_url, output_path)

print("Getting text from video...")

client = OpenAI(api_key="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")

audio_file= open(r".\Downloaded Videos\{}.mp4".format(video_title), "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)

print(transcript)