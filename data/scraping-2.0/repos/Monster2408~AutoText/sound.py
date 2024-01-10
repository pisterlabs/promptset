# -*- coding: utf8 -*-
from openai import OpenAI
import os
import ffmpeg 

file_name = "movie_split_"

def sound_to_text(api_key: str, encode_dir: str):
    client = OpenAI(api_key=api_key)

    dir_path = encode_dir + os.sep + "tmp"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    if not os.path.exists(encode_dir):
        os.makedirs(encode_dir)

    # dir_pathの中のファイルを取得
    num: int = 0

    while True:
        file_path = dir_path + file_name + str(num) + ".wav"
        if not os.path.exists(file_path):
            break
        print(file_path)
        file_size_byte = os.path.getsize(file_path)

        file_size_mb = file_size_byte / 1024 / 1024
        if (file_size_mb > 25) :
            print("ファイルサイズが25MBを超えています。")
            continue

        audio_file= open(file_path, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        print(transcript)
        
        # encode_dirにtxtファイルを作成
        encode_file_path = encode_dir + file_name + str(num) + ".txt"
        with open(encode_file_path, mode='w') as f:
            f.write(transcript.text)

        
        num += 1
        
        print("")

def split(file_path: str):
    file_size_byte = os.path.getsize(file_path)
    file_size_mb = file_size_byte / 1024 / 1024

    if file_size_mb >= 25:
        num = int(file_size_mb / 25)
        num =+ 1
    else:
        num = 1

    print("分割数: " + str(num))

    split_num = num

    # 音声ファイルの長さを取得
    probe = ffmpeg.probe(file_path)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    duration = float(audio_stream['duration'])
    print("duration: " + str(duration))

    # 分割する時間を計算
    split_time = duration / split_num
    print("split_time: " + str(split_time))

    # 分割する時間を指定して分割
    for i in range(split_num):
        print("split: " + str(i))
        ffmpeg.input(file_path, ss=i*split_time, t=split_time).output("movie_split_" + str(i) + ".wav").run()

    # 最後のファイルを分割
    print("split: " + str(split_num))

    ffmpeg.input(file_path, ss=split_num*split_time).output("movie_split_" + str(split_num) + ".wav").run()