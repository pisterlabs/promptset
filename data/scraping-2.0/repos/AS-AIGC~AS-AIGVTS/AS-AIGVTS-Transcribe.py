#!/usr/bin/env python3

from pytube import YouTube    # library for downloading YouTube videos
from pydub import AudioSegment    # library for working with audio files
import pysrt
import whisper    # library for sending and receiving messages over a network
from whisper.utils import get_writer
import openai    # library for working with the OpenAI API
import os, sys, traceback
from datetime import datetime
from tqdm import tqdm
from googletrans import Translator, LANGUAGES
import config

# Retrieve configurations from the config file
youtube_list = config.YouTube_List
PREFIX = config.PREFIX
LANGUAGES = config.LANGUAGES

# Function to slice audio files
def slice_audio(audio_file, filename, offset):
    audio_length = audio_file.duration_seconds
    minutes_duartion = int(audio_length // 60)

    one_minutes = 1 * 60 * 1000
    start = offset * one_minutes
    end = audio_length if start == minutes_duartion else (offset+1) * one_minutes
    sliced_audio = audio_file[start:end]
    sliced_audio.export(filename, format="mp3")

# Function to concatenate srt files
def concatenate_srt_file(main, sliced_part, offset):
    main_subtitles = pysrt.open(main)
    sliced_part_subtitles = pysrt.open(sliced_part)
    sliced_part_subtitles.shift(minutes=offset)
    main_subtitles_length = len(main_subtitles)
    for subtitle in sliced_part_subtitles:
        subtitle.index += main_subtitles_length
        main_subtitles.append(subtitle)
    main_subtitles.save(main, encoding='utf-8')

# Function to translate srt files using googletrans
def translate_srt_file_by_googletrans(lang, sliced_part_srt, sliced_part_subtitle_srt):
    translator = Translator()
    subtitles = pysrt.open(sliced_part_srt)
    for subtitle in subtitles:
        translated_subtitle = translator.translate(text=subtitle.text, dest=lang)
        subtitle.text = translated_subtitle.text
    subtitles.save(sliced_part_subtitle_srt, encoding='utf-8')

# Iterate over the items in the youtube_list dictionary
for k, v in youtube_list.items():
    start_time = datetime.now()
    try:
        print("processing ", v)
        yt = YouTube(f"https://www.youtube.com/watch?v={v}", use_oauth=True)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(output_path="/tmp/", filename=f"audio_{k}")
        audio_file = AudioSegment.from_file(f"/tmp/audio_{k}")
        audio_file.export(f"/tmp/audio_{k}.mp3", format="mp3")
        mp3_file = AudioSegment.from_file(f"/tmp/audio_{k}.mp3", 'mp3')

        mp3_duration_minutes = int(mp3_file.duration_seconds // 60)
        seconds = round(mp3_file.duration_seconds - mp3_duration_minutes * 60, 2)
        model = whisper.load_model("small")

        with open(f"{PREFIX}{k}.txt", 'w') as out_txt, open(f"{PREFIX}{k}.srt", 'w') as out_srt:

            for lang in LANGUAGES:
                out_offset = open(f"{PREFIX}{k}_{lang}.srt",'w')

            for offset in tqdm(range(mp3_duration_minutes + 1)):
                fname_offset = f"{k}_{offset}_{offset+1}"
                fname_offset_mp3 = f"/tmp/{fname_offset}.mp3"
                slice_audio(mp3_file, fname_offset_mp3, offset)

                result = model.transcribe(fname_offset_mp3, fp16=False)
                txt_writer = get_writer("txt","/tmp/")
                txt_writer(result,f"{fname_offset}.txt")
                srt_writer = get_writer("srt","/tmp/")
                srt_writer(result,f"{fname_offset}.srt")

                with open(f"/tmp/{fname_offset}.txt") as infile:
                    out_txt.write(infile.read() + " ")

                concatenate_srt_file(f"{PREFIX}{k}.srt", f"/tmp/{fname_offset}.srt", offset)

                for lang in LANGUAGES:
                    translate_srt_file_by_googletrans(lang, f"/tmp/{fname_offset}.srt", f"/tmp/{fname_offset}_{lang}.srt")
                    concatenate_srt_file(f"{PREFIX}{k}_{lang}.srt", f"/tmp/{fname_offset}_{lang}.srt", offset)
                    if os.path.exists(f"/tmp/{fname_offset}_{lang}.srt"):
                        os.remove(f"/tmp/{fname_offset}_{lang}.srt")

                # Removing temp files
                if os.path.exists(f"/tmp/{fname_offset}.txt"):
                    os.remove(f"/tmp/{fname_offset}.txt")
                if os.path.exists(f"/tmp/{fname_offset}.srt"):
                    os.remove(f"/tmp/{fname_offset}.srt")
                if os.path.exists(f"/tmp/{fname_offset}.mp3"):
                    os.remove(f"/tmp/{fname_offset}.mp3")

        # Removing temp files
        if os.path.exists("/tmp/audio_" + k + ".mp3"):
            os.remove("/tmp/audio_" + k + ".mp3")
        if os.path.exists("/tmp/audio_" + k):
            os.remove("/tmp/audio_" + k)

    except Exception as ex:
        print(f"Exception type : {type(ex).__name__}")
        print(f"Exception message : {str(ex)}")
        print("Stack trace :")
        traceback.print_exc()

    end_time = datetime.now()
    delta_time = end_time - start_time
    print(f"{os.path.basename(__file__)},{k},{v},{delta_time.total_seconds()}")
