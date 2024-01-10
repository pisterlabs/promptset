import os
import sys
import openai
import subprocess
# from pytube import YouTube
from yt_dlp import YoutubeDL

# not sure how to do this properly yet: https://platform.openai.com/docs/guides/speech-to-text/quickstart

def download_video_by_pytube(video_id):
    video_url = f'http://youtube.com/watch?v={video_id}'
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(filename="video.mp3", output_path=".", skip_existing=False)
        print(f"Audio downloaded from {video_url}")
    except Exception as e:
        print(f"Error downloading audio from {video_url}: {e}")

    with YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(URLS)

def download_video_by_yt_dlp(video_id):
    video_url = f'http://youtube.com/watch?v={video_id}'
    URLS = [video_url]
    try:
        ydl_opts = {
            "outtmpl": './video.%(ext)s',
            'format': 'm4a/bestaudio/best',
            # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
            'postprocessors': [{  # Extract audio using ffmpeg
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }]
        }

        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(URLS)
    except Exception as e:
        print(f"Error downloading audio from {video_url}: {e}")


def transcribe_audio():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    filename = "./video.mp3"
    file_size = os.path.getsize(filename) / (1024 * 1024)
    if file_size > 24:
        # Extract the current bitrate from the file
        ffprobe_cmd = f"ffprobe -v error -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 {filename}"
        current_bitrate = int(subprocess.check_output(ffprobe_cmd, shell=True))

        # Calculate the new bitrate (half of the current bitrate)
        new_bitrate = current_bitrate * 24 // file_size
        print(f"[Pre-transcribe] File size {file_size} too large, lowering bitrate from {current_bitrate} to {new_bitrate}")
        # Create the new file name
        new_filename = f"./reduced_video.mp3"

        # Execute the ffmpeg command to reduce the bitrate
        ffmpeg_cmd = f"ffmpeg -i {filename} -b:a {new_bitrate} {new_filename}"
        subprocess.run(ffmpeg_cmd, shell=True)
        audio_file = open(new_filename, "rb")
    else:
        audio_file = open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="srt")
    os.remove(filename)
    if file_size > 24:
        os.remove(new_filename)
    return transcript

def main(video_id, download_method):
    try:
        if(download_method == "pytube"):
            download_video_by_pytube(video_id)
        else:
            download_video_by_yt_dlp(video_id)
        transcript = transcribe_audio()

        with open(f"./static/src/subtitle/{video_id}.srt", 'w', encoding='utf-8') as f:
            f.write(transcript)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    video_id = sys.argv[1]
    download_method = sys.argv[2]
    main(video_id, download_method)