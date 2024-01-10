from datetime import timedelta
from math import floor
from pathlib import Path
from random import randint, randrange
import random
from moviepy.video.fx import resize, crop
from moviepy.audio.fx import multiply_volume
from moviepy import *
from typing import Tuple
from configuration import Configuration
from openai_interface import OpenAiInterface

from reddit_requests import Post

config = Configuration()

POSSIBLE_FILE_ENDINGS = (".mp4", ".webm", ".mkv", ".ogv", ".mpeg", ".avi", ".mov")

CHARS_PER_SECOND = (10000 / 10.5) / 60


def select_background_video(min_length: int) -> Tuple[VideoClip, str]:
    possible_videos = [
        p.resolve()
        for p in Path(config.background_videos_dir).glob("**/*")
        if p.suffix in POSSIBLE_FILE_ENDINGS
    ]

    clip: VideoClip | None = None
    background_video_credit: str | None = None

    random.shuffle(possible_videos)
    for video in possible_videos:
        clip = VideoFileClip(video)
        if min_length <= clip.duration:
            selected_file = video
            background_video_credit = str(selected_file).split("\\")[-2]
            print(f"selected {selected_file} as background video")
            start_time = random.random() * (clip.duration - min_length)
            end_time = start_time + min_length
            print(
                f"using background video time between {start_time:.2f}s and {end_time:.2f}s out of {clip.duration:.2f}s"
            )
            clip = clip.subclip(start_time, end_time)
            clip = clip.afx(multiply_volume, config.background_video_volume)  # type: ignore
            break

    if clip == None or background_video_credit == None:
        raise Exception(f"No suitable background video found for duration {min_length}")

    return (clip, background_video_credit)


def crop_to_center_and_resize(clip: VideoClip, to_resolution: Tuple[int, int]):
    new_aspect_ratio: float = to_resolution[0] / to_resolution[1]
    x1 = (clip.size[0] - (clip.size[1] * new_aspect_ratio)) // 2
    x2 = (clip.size[0] + (clip.size[1] * new_aspect_ratio)) // 2
    y1 = 0
    y2 = clip.size[1]

    clip = clip.fx(crop, x1=x1, x2=x2, y1=y1, y2=y2)
    clip = clip.fx(resize, to_resolution)
    return clip


def generate_intro_clip(post: Post, resolution: Tuple[int, int]) -> VideoClip:
    openaiinterface = OpenAiInterface()
    intro_text = openaiinterface.generate_text_without_context(
        config.intro_prompt,
        post.title + "\n" + post.selftext,
    )
    openaiinterface.generate_mp3(intro_text, f"tmp/{post.post_id}-audio-intro.mp3")

    intro_clip: VideoClip = TextClip(
        config.intro_header + "\n" + post.title,
        size=(resolution[0] * 0.8, 0),
        color=config.video_font_color,
        font=config.video_font,
        font_size=config.video_font_size,
        method="caption",
        stroke_color=config.video_font_stroke_color,
        stroke_width=config.video_font_stroke_width,
        align="center",
    )
    audio_clip = AudioFileClip(f"tmp/{post.post_id}-audio-intro.mp3")
    intro_clip = intro_clip.with_duration(audio_clip.duration + 1)
    intro_clip = intro_clip.with_audio(audio_clip)

    print(f"Created intro with duration {intro_clip.duration}s")

    return intro_clip


def generate_outro_clip(post: Post, resolution: Tuple[int, int]) -> VideoClip:
    openaiinterface = OpenAiInterface()
    outro_text = openaiinterface.generate_text_without_context(
        config.outro_prompt,
        post.title + "\n" + post.selftext,
    )
    openaiinterface.generate_mp3(outro_text, f"tmp/{post.post_id}-audio-outro.mp3")

    outro_clip: VideoClip = TextClip(" ")
    audio_clip = AudioFileClip(f"tmp/{post.post_id}-audio-outro.mp3")
    outro_clip = outro_clip.with_duration(audio_clip.duration + 1)
    outro_clip = outro_clip.with_audio(audio_clip)

    print(f"Created outro with duration {outro_clip.duration}s")
    return outro_clip


def check_if_valid_post(
    post_id: str,
    post_title: str,
    text_to_check: str,
    approx_video_duration: timedelta | None = None,
    min_duration: timedelta | None = None,
) -> bool:
    with open("config/already_posted.txt", "r") as file:
        already_posted_ids = file.read().splitlines()
    if post_id in already_posted_ids:
        print(f"Post {post_id} has already been posted")
        return False

    filter_word_in_title = ["update:", "(update)", "[update]"]
    for word in filter_word_in_title:
        if word in post_title.lower():
            print(f"Post {post_id} is an update")
            return False

    if post_title.lower().startswith("update"):
        print(f"Post {post_id} is an update")
        return False

    print(approx_video_duration)
    if approx_video_duration != None and not is_approx_duration(
        text_to_check, approx_video_duration
    ):
        print(
            f"Post {post_id} duration is not approximatly {approx_video_duration} long."
        )
        return False

    if min_duration != None and not is_min_duration(text_to_check, min_duration):
        print(f"Post {post_id} duration is not over {min_duration} long.")
        return False

    print(f"Post {post_id} is valid")
    return True


def is_max_duration(text: str, max_duration: timedelta) -> bool:
    text_duration = len(text) / CHARS_PER_SECOND
    if max_duration.total_seconds() < text_duration:
        return False
    return True


def is_min_duration(text: str, min_duration: timedelta) -> bool:
    text_duration = len(text) / CHARS_PER_SECOND
    if min_duration.total_seconds() > text_duration:
        return False
    return True


def is_between_durations(
    text: str, min_duration: timedelta, max_duration: timedelta
) -> bool:
    if is_min_duration(text, min_duration) and is_max_duration(text, max_duration):
        return True
    return False


def is_approx_duration(text: str, approx_duration: timedelta) -> bool:
    upper_bound = approx_duration + (approx_duration * config.tolerated_duration_offset)
    lower_bound = approx_duration - (approx_duration * config.tolerated_duration_offset)

    if is_between_durations(text, lower_bound, upper_bound):
        return True
    return False
