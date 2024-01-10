import os
import sys
from pytube import YouTube
import openai
from moviepy.editor import *

def download_audio(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    output_path = os.path.join(os.getcwd(), 'audio.mp4')
    audio_stream.download(output_path)

    # Convert the audio to mp3 using moviepy (requires ffmpeg)
    audio_clip = AudioFileClip(get_first_file_name_in_directory(output_path))
    output_mp3_path = os.path.join(os.getcwd(), 'audio.mp3')
    audio_clip.write_audiofile(output_mp3_path, logger=None)
    audio_clip.close()
    os.remove(get_first_file_name_in_directory(output_path))

    return output_mp3_path

def get_first_file_name_in_directory(directory_path):
    try:
        for file_name in os.listdir(directory_path):
            if file_name.lower().endswith(".mp4"):
                full_file_path = os.path.join(directory_path, file_name)
                base_name = os.path.basename(file_name)  # Get the file name only
                name_without_extension, _ = os.path.splitext(base_name)  # Remove the extension
                print(name_without_extension, '\n')
                return full_file_path
    except Exception as e:
        raise Exception("Failed to get the first file name: " + str(e))


system_prompt = "You are a summary writer. Write the summary of the transcript of the following audio file which explains in detail the topic and information covered in it.  Write detailed summary and explanation on the core information or discussion of the video. Do not refer to the text as a transcript and directly give a summary as if a person was explaining it. Keep it under 1900 characters. Keep it in active noise as if the speaker is summarizing the text." 

def generate_corrected_transcript(temperature, system_prompt):
    audio_file= open("audio.mp3", "rb")
    responseText = openai.Audio.transcribe("whisper-1", audio_file)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": responseText['text']
            }
        ]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    openai.api_key = "Insert your own key"
    if len(sys.argv) < 2:
        print("Please provide a YouTube URL as a command line argument.")
        sys.exit()

    youtube_url = sys.argv[1]
    audio_file = download_audio(youtube_url)
    summary_text = generate_corrected_transcript(0, system_prompt)
    print(summary_text)
