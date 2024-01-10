import shlex
import subprocess
from uuid import uuid4

import openai
import requests
import os
import urllib.request
import ssl
import models
import re
import ffmpeg

from config import settings

ssl._create_default_https_context = ssl._create_unverified_context

openai.api_key = settings.OPENAI_API_KEY


def create_chat_completion(system, message):
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    )
    return chat_completion.choices[0].message.content


def get_stock_video(query: str):
    search_url = f"https://api.pexels.com/videos/search?"
    params = {
        "query": query,
        "orientation": "portrait",
        "size": "medium",
    }
    headers = {"Authorization": settings.STOCK_VIDEO_API_KEY}
    res = requests.get(search_url, params=params, headers=headers)
    result = res.json()
    d = result["videos"][0]["id"]

    return result["videos"][0]["id"]


def save_video(video_id, output_path):
    video_url = f"https://www.pexels.com/download/video/{video_id}"
    params = {"h": "1920", "w": "1080"}

    video = requests.get(video_url, params=params).content

    print(video_url)
    with open(output_path, "wb") as f:
        f.write(video)

    return output_path


# todo: 예외처리 of subprocess
def combine_audio_and_video(video_path, audio_path, subtitle_path, output_path):
    command = shlex.split(
        f"ffmpeg \
            -stream_loop -1 -i {video_path} -i {audio_path} \
            -vf \"subtitles={subtitle_path}:force_style='OutlineColour=&H80000000,BorderStyle=4,BackColour=&H80000000 \
            Outline=0,Shadow=0,MarginV=25,Fontname=Arial,Fontsize=12,Alignment=2'\" \
            -c:v mpeg2video -qscale:v 2 \
            -map 0:v -map 1:a \
            -shortest -fflags shortest -max_interleave_delta 100M \
            -y {output_path}"
    )

    subprocess.run(command)


def create_image_keyword(message):
    keywords = create_chat_completion(
        "Please recommend three short keywords in English to draw background image that expresses message in dall-e. "
        "separated by ',' and please no numbering, only keywords and please don't contain any other words",
        message,
    )

    return keywords


def create_speech_from_text(message, file_name, voice_code):
    client_id = settings.TTS_CLIENT_ID
    client_secret = settings.TTS_CLIENT_SECRET

    encrypted_text = urllib.parse.quote(message)
    data = f"speaker={voice_code}&volume=0&speed=0&pitch=0&format=mp3&text={encrypted_text}"
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"

    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    request.add_header("X-NCP-APIGW-API-KEY", client_secret)

    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    res_code = response.getcode()

    if res_code != 200:
        raise Exception("TTS Error Code:" + res_code)

    response_body = response.read()
    with open(file_name, "wb") as out:
        out.write(response_body)

    return file_name


def split_string(s):
    sentences = re.split("([.!?])", s)
    sentences = ["".join(i) for i in zip(sentences[::2], sentences[1::2])]
    sentences = [sentence.rstrip() for sentence in sentences]

    return sentences


def concat_videos(input_video_path_list, output_video_path: str):
    concat_string = "|".join(input_video_path_list)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            f"concat:{concat_string}",
            "-r",
            "30",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            output_video_path,
        ]
    )

    return output_video_path


def composite_videos(uuid, n: int):
    file_names = [f"static/{uuid}/output{idx}.ts" for idx in range(n)]
    concat_videos(file_names, f"static/{uuid}/result.mp4")


def merge_audio_files(audio_files: list, output_path: str):
    command = shlex.split(
        f'ffmpeg -y {"".join([f"-i {audio_file} " for audio_file in audio_files])}'
        f'-filter_complex "{"".join([f"[{i}:a]" for i in range(len(audio_files))])}'
        f'concat=n={len(audio_files)}:a=1:v=0" {output_path}'
    )

    subprocess.run(command)

    return output_path


def convert_to_srt_format(seconds):
    # 초를 시, 분, 초로 변환
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60

    # 소수점 아래 3자리까지만 표시
    seconds = round(seconds, 3)

    # 소수점 아래 3자리를 밀리초로 변환
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    # 시, 분, 초, 밀리초를 '00:00:06,168' 형태의 문자열로 변환
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# ffmpeg -i audio.mp3 -i audio.mp3  -filter_complex "[0:a][1:a]concat=n=2:a=1:v=0" output.mp3
def make_subtitle_from_audio(audio_files: list, messages: list, output_path: str):
    durations = [
        float(ffmpeg.probe(audio_file)["format"]["duration"])
        for audio_file in audio_files
    ]
    summed_durations = [0] + [duration for duration in durations]

    for i in range(1, len(summed_durations)):
        summed_durations[i] += summed_durations[i - 1]

    srt_str = ""
    for i in range(1, len(summed_durations)):
        srt_str += (
            f"{i}\n"
            f"{convert_to_srt_format(summed_durations[i - 1])} --> {convert_to_srt_format(summed_durations[i])}\n"
            f"{messages[i-1]}\n\n"
        )

    with open(output_path, "w") as f:
        f.write(srt_str)

    return output_path


def make_video(keywords, tts_id, db):
    voice_code = (
        db.query(models.TTSInfo).filter(models.TTSInfo.id == tts_id).first().voice_code
    )
    print("making start")

    # m = create_chat_completion(
    #     system="한국어 대본을 작성해줘. 각 단계의 이름은 쓰지마. Make a 200-character script in YouTube Shorts video style for "
    #     "Korean scripts. Write in the introduction, development, turn, and conclusion structure, "
    #     "and separate each step by '\n'. Separate each sentence with '.'.",
    #     message=keywords,
    # )
    m = "안녕하세요! 오늘은 졸업 프로젝트에서 좋은 학점을 받는 법에 대해 알려드리겠습니다. 첫 번째로, 프로젝트 주제를 잘 선택하세요. 주제는 흥미로운 동시에 실용적이어야 합니다. 두 번째로, 신중하게 계획을 세워주세요. 프로젝트 일정과 목표니다. 네 번째로, 분석과 실험을 철저히 진행하세요. 데이터를 정확하게 분석하고, 필요한 실험을 진행하는 것이 중요합니다. 마지막으로, 결과를 명확하게 발표하고 정리하세요. 프로젝트의 결과물을 잘 보여주고, 핵심 내용을 명확하게 전달하"
    print(m)

    uuid = uuid4()
    db_video = models.Video(id=str(uuid), keyword=keywords, script=m)
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    os.makedirs(f"static/{uuid}")

    paragraphs = [p for p in m.split("\n") if p != ""]

    for idx, paragraph in enumerate(paragraphs):
        messages = split_string(paragraph)
        image_keywords = create_image_keyword(paragraph)
        print(image_keywords)

        # create tts mp3 files
        audio_files = [
            create_speech_from_text(m, f"static/{uuid}/output{idx}-{i}.mp3", voice_code)
            for i, m in enumerate(messages)
        ]
        # create srt file from mp3 files
        subtitle_path = make_subtitle_from_audio(
            audio_files, messages, f"static/{uuid}/subtitle{idx}.srt"
        )
        # merge audio files
        audio_path = merge_audio_files(audio_files, f"static/{uuid}/output{idx}.mp3")

        combine_audio_and_video(
            save_video(
                get_stock_video(image_keywords), f"static/{uuid}/bg_video{idx}.mp4"
            ),
            audio_path,
            subtitle_path,
            f"static/{uuid}/output{idx}.ts",
        )
        db_sub_video = models.SubVideo(
            script=paragraph, keyword=image_keywords, full_video_id=str(uuid), index=idx
        )
        db.add(db_sub_video)
        db.commit()
        db.refresh(db_sub_video)

    composite_videos(uuid, len(paragraphs))

    return uuid


def get_tts_infos(db):
    return db.query(models.TTSInfo).all()
