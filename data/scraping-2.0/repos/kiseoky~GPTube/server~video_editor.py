import openai
import requests
import os

import models
from config import settings

from video_maker import (
    split_string,
    composite_videos,
    create_speech_from_text,
)

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

from moviepy.config import change_settings

change_settings(
    {"IMAGEMAGICK_BINARY": "/opt/homebrew/Cellar/imagemagick/7.1.1-15/bin/convert"}
)


openai.api_key = settings.OPENAI_API_KEY


def generate_images(keyword: str, n, size):
    image_resp = openai.Image.create(
        prompt=f"{keyword}, high quality, simple background", n=n, size="1024x1024"
    )
    img_urls = [x.url for x in image_resp.data]

    return img_urls


def show_sub_videos(uuid: str, db):
    return db.query(models.Video).filter(models.Video.id == uuid).first().sub_videos


def get_sub_video_info(uuid, index, db):
    return {
        "keyword": db.query(models.Video)
        .filter(models.Video.id == uuid)
        .first()
        .sub_videos[index]
        .keyword,
        "selected_tts": db.query(models.Video)
        .filter(models.Video.id == uuid)
        .first()
        .sub_videos[index]
        .tts_id,
    }


def apply_change_to_video(uuid, index, db):
    sub_video = (
        db.query(models.Video).filter(models.Video.id == uuid).first().sub_videos[index]
    )
    messages = split_string(sub_video.script)
    n = len(messages)
    audio_paths = [f"static/{uuid}/{index}-output{i}.mp3" for i in range(n)]
    output_path = f"static/{uuid}/output{index}.ts"
    # add_static_image_to_audio(
    #     messages=messages,
    #     image_path=f"static/{uuid}/img{index}.png",
    #     audio_paths=audio_paths,
    #     output_path=output_path,
    # )


def edit_image(uuid: str, index: int, url: str, db):
    img_data = requests.get(url).content
    file_name = f"static/{uuid}/img{index}.png"
    with open(file_name, "wb") as handler:
        handler.write(img_data)

    apply_change_to_video(uuid, index, db)


def edit_audio(uuid, index, tts_id, db):
    voice_code = (
        db.query(models.TTSInfo).filter(models.TTSInfo.id == tts_id).first().voice_code
    )
    sub_video = (
        db.query(models.Video).filter(models.Video.id == uuid).first().sub_videos[index]
    )
    script = sub_video.script

    messages = split_string(script)

    for i, message in enumerate(messages):
        create_speech_from_text(
            message, f"static/{uuid}/{index}-output{i}.mp3", voice_code
        )

    sub_video.tts_id = tts_id
    db.commit()
    apply_change_to_video(uuid, index, db)


def merge_video(uuid: str, db):
    n = len(db.query(models.Video).filter(models.Video.id == uuid).first().sub_videos)

    composite_videos(uuid, n)
