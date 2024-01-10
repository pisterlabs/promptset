import openai
from videoproc import youtube_preprocess
from chunking import chunk_by_size

openai.api_key="YOURAPIKEY"

# Open the file and read the URLs
with open('youtube_urls.txt', 'r') as file:
    urls = file.readlines()

for url in urls:
    url = url.strip()  # remove newline characters

    # Extract video ID from URL to use as filename
    video_id = url.split('=')[-1]
    filename = f"{video_id}.txt"

    audio_file = youtube_preprocess(url)

    no_of_chunks = chunk_by_size(audio_file)

    for i in range(no_of_chunks):
        print(f"process_chunks/chunk{i}.wav")
        curr_file = open(f"process_chunks/chunk{i}.wav", "rb")
        transcript = openai.Audio.transcribe("whisper-1", curr_file, prompt="""This is either an interview or speech by Micah White, the co-creator of Occupy Wall Street, and author of The End of Protest: A New Playbook for Revolution.""")

        # Write transcript to a new file named after the video ID
        with open(filename,"a") as f:
            f.write(transcript["text"])
