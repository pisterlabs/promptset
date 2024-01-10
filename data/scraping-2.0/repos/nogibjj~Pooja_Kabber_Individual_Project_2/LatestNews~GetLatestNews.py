import re
import scrapetube
from pytube import YouTube
import os
import whisper
import openai
import warnings

warnings.filterwarnings("ignore")

channels = {
    "cnn": "UCupvZG-5ko_eiXAupbDfxWw",
    "bbc": "UC16niRr50-MSBwiO3YDb3RA",
    "aljazeera": "UCNye-wNBqNL5ZzHSJj3l8Bg",
    "msnbc": "UCaXkIU1QidjPwiAYu6GcHjg",
    "ndtv": "UCZFMm1mMw0F81Z37aaEzTUA",
}

model = whisper.load_model("tiny.en")
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_news(channel, length, n, title):

    videos = scrapetube.get_channel(channels[channel])

    i = 0

    for video in videos:
        link = "https://www.youtube.com/watch?v=" + str(video["videoId"])

        yt = YouTube(link)

        if yt.length < 500:

            # extract only audio
            audio_only = yt.streams.filter(only_audio=True).first()

            # download the file
            out_file = audio_only.download()

            new_file = f"audio_file{i}" + ".mp3"
            os.rename(out_file, new_file)

            if(title):
                print(f"TITLE: {yt.title}")

            result = model.transcribe(new_file)
            news_text = result["text"]

            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=f"{news_text}\n\nTl;dr",
                temperature=0.8,
                max_tokens=length,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            
            response_text = response["choices"][0]["text"]
            if(response_text[0:3] == ": ") or (response_text[0:3] == ":\n"):
                response_text = response_text[3:]
            print(response_text + "\n")

            i = i + 1

        if i > n:
            break
