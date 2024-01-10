import os
import subprocess
import shutil
import whisper
import time
import openai
from pytube import YouTube


class Transcriber:
    def __init__(self, model_size, openai_key=None):
        if openai_key:
            openai.api_key = openai_key
        else:
            self.model = whisper.load_model(model_size)

    def download_audio(self, link):
        # this function downloads the video from YouTube, extracts audio and saves it
        os.makedirs("./download/", exist_ok=True)
        os.makedirs("./Data/", exist_ok=True)
        path = "./download/"
        yt = YouTube(link)
        audio = yt.streams.filter(only_audio=True)[0]
        audio.download(path)
        file_name = audio.default_filename
        aud_name = "./Data/Audio.m4a"
        # convert to mp3
        subprocess.run(["ffmpeg", "-i", os.path.join(path, file_name),
                        "-c:a", "copy", "-y", os.path.join(aud_name)])
        # remove the video file
        os.remove(os.path.join(path, file_name))
        self.segment_audio()

    def extract_audio_gdrive(self, gdrive_path):
        # this function downloads the video from GDrive, extracts audio and saves it
        os.makedirs("./download/", exist_ok=True)
        os.makedirs("./Data/", exist_ok=True)
        aud_name = "./Data/Audio.m4a"
        # convert to mp3
        subprocess.run(["ffmpeg", "-i", gdrive_path, "-map", "0:a",
                        "-c", "copy", "-y", os.path.join(aud_name)])
        self.segment_audio()

    def segment_audio(self):
        # Split the audio into 10 minute length chunks, so it is easier to process
        os.makedirs("./Data/Chunks/", exist_ok=True)
        subprocess.run(["ffmpeg", "-i", "./Data/Audio.m4a", "-f", "segment",
                        "-segment_time", "600", "-c", "copy", "-y", "./Data/Chunks/%03d.m4a"])

    def transcribe(self, file):
        return self.model.transcribe(file)["segments"]

    def transcribe_api(self, file):
        af = open(file, "rb")
        return openai.Audio.transcribe("whisper-1", file=af, temperature=0.0)

    def write_api_result(self, result, vid_name, ind):
        # save results
        os.makedirs("./Results/{}/".format(vid_name), exist_ok=True)
        with open("./Results/{}/{:03d}-{:03d}_noTimestamp.txt".format(vid_name, ind * 10, ind * 10 + 10), "w") as f:
            f.write(result["text"])
        pass

    def write_result(self, result, vid_name, ind):
        txt = ""
        txt_noTime = ""
        for entry in result:
            start_time = time.strftime('%H:%M:%S', time.gmtime(entry["start"]))
            end_time = time.strftime('%H:%M:%S', time.gmtime(entry["end"]))
            txt += "{} - {}: {} \n".format(start_time, end_time, entry["text"])
            txt_noTime += entry["text"]

        # save results
        os.makedirs("./Results/{}/".format(vid_name), exist_ok=True)
        with open("./Results/{}/{:03d}-{:03d}.txt".format(vid_name, ind * 10, ind * 10 + 10), "w") as f:
            f.write(txt)
        with open("./Results/{}/{:03d}-{:03d}_noTimestamp.txt".format(vid_name, ind * 10, ind * 10 + 10), "w") as f:
            f.write(txt_noTime)

    def clear(self):
        shutil.rmtree("./download/")
        shutil.rmtree("./Data/")