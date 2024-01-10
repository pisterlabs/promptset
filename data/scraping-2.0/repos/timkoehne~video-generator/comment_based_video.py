from datetime import timedelta
from random import randrange
from typing import Literal, Tuple

from moviepy import *
from configuration import Configuration
from video_utils import (
    CHARS_PER_SECOND,
    check_if_valid_post,
    crop_to_center_and_resize,
    generate_intro_clip,
    generate_outro_clip,
    select_background_video,
)
from openai_interface import OpenAiInterface
from reddit_requests import Comment, Post, PostSearch
from text_processing import split_text_to_max_x_chars

config = Configuration()


def calculate_font_size(text: str) -> Tuple[list[str], list[int]]:
    # TODO implement text_wall_font_size configuration option

    text_parts: list[str] = split_text_to_max_x_chars(text, 550)
    font_sizes: list[int] = []

    for part in text_parts:
        if len(part) > 400:
            font_sizes.append(45)
        elif len(part) > 200:
            font_sizes.append(50)
        else:
            font_sizes.append(70)

    return (text_parts, font_sizes)


def generate_single_comment_clip(
    post: Post,
    text_parts: list[str],
    font_sizes: list[int],
    resolution: Tuple[int, int],
    index: int,
):
    openaiinterface = OpenAiInterface()
    comment_clip_parts: list[VideoClip] = []

    for i, part in enumerate(text_parts):
        openaiinterface.generate_mp3(
            part, f"tmp/{post.post_id}-audio-{index}-part-{i}.mp3"
        )
        clip_part: VideoClip = TextClip(
            part,
            size=(resolution[0] * 0.8, 0),
            color=config.text_wall_font_color,
            font=config.text_wall_font,
            font_size=font_sizes[i],
            method="caption",
            stroke_color=config.text_wall_font_stroke_color,
            stroke_width=config.text_wall_font_stroke_width,
            align="center",
        )

        audio_clip = AudioFileClip(f"tmp/{post.post_id}-audio-{index}-part-{i}.mp3")
        clip_part = clip_part.with_duration(audio_clip.duration + 1)
        clip_part = clip_part.with_audio(audio_clip)
        comment_clip_parts.append(clip_part)

    comment_clip = concatenate_videoclips(comment_clip_parts)
    return comment_clip


def generate_comments_clip(post: Post, resolution: Tuple[int, int]) -> VideoClip:
    text_clips: list[VideoClip] = []
    intro: VideoClip = generate_intro_clip(post, resolution)
    outro: VideoClip = generate_outro_clip(post, resolution)

    comments: list[Comment] = post.get_good_comments()
    print(f"There are {len(comments)} good comments")

    for index, comment in enumerate(comments):
        print(comment)

        text_parts, font_sizes = calculate_font_size(comment.body)
        if len(text_parts) > 1:
            print(f"Splitting Comment into {len(text_parts)} parts")

        comment_clip = generate_single_comment_clip(
            post, text_parts, font_sizes, resolution, index
        )
        text_clips.append(comment_clip)
    combined_text_video: VideoClip = concatenate_videoclips(text_clips)
    combined_text_video = concatenate_videoclips([intro, combined_text_video, outro])
    combined_text_video = combined_text_video.with_position("center")
    return combined_text_video


def find_comment_post(
    timeframe: Literal["day", "week", "month", "year", "all"],
    listing: Literal["controversial", "best", "hot", "new", "random", "rising", "top"],
    subreddit_list: list[str],
    approx_video_duration: timedelta | None = None,
):
    maxAttempts = 50
    while True:
        subreddit = subreddit_list[randrange(0, len(subreddit_list))]
        search = PostSearch(subreddit, listing, timeframe)

        if len(search.posts) < 1:
            continue
        selected_post = search.posts[randrange(0, len(search.posts))]

        if approx_video_duration != None:
            good_comments = selected_post.get_good_comments(
                num_chars_to_limit_comments=int(
                    approx_video_duration.total_seconds() * CHARS_PER_SECOND
                )
            )
        else:
            good_comments = selected_post.get_good_comments()

        comments_combined = " ".join([c.body for c in good_comments])

        valid = check_if_valid_post(
            selected_post.post_id,
            selected_post.title,
            comments_combined,
            approx_video_duration,
        )

        if valid:
            break
        else:
            maxAttempts -= 1
            if maxAttempts <= 0:
                raise Exception(f"No valid post found in {maxAttempts} attempts.")
    return selected_post
