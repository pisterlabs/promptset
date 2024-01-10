import os
import random
import sys
from typing import Tuple
import cv2
from moviepy import *
from PIL import Image
import numpy as np

from configuration import Configuration
from db_access import DB_Controller
from openai_interface import OpenAiInterface
from reddit_requests import Post

config = Configuration()


class NoSuitablePersonException(Exception):
    pass


class Rectangle:
    def __init__(
        self,
        top_left: Tuple[float, float],
        top_right: Tuple[float, float],
        bottom_left: Tuple[float, float],
        bottom_right: Tuple[float, float],
    ) -> None:
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        pass

    def width(self):
        return self.top_right[0] - self.top_left[0]

    def height(self):
        return self.bottom_left[1] - self.top_left[1]

    def aspect_ratio(self):
        return self.width() / self.height()


def get_bordered_pil_image(path: str, edge_size: int):
    src = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    alpha = cv2.split(src)[3]

    # scale so it is roughly the same at every resolution
    edge_size = int(edge_size * len(alpha[0]) / 1000)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(alpha, kernel, iterations=edge_size)
    output_image = cv2.merge((dilated, dilated, dilated, dilated))
    pil_image = Image.fromarray(output_image)

    return pil_image


def find_thumbnail_person(bounds: Rectangle, keywords: list[str]) -> Image.Image:
    db_controller = DB_Controller("database.db")
    candidates = db_controller.find_images_with_tags(keywords)
    random.shuffle(candidates)
    for candidate in candidates:
        person = Image.open(os.path.join(config.thumbnail_image_dir, candidate))
        aspect_ratio = person.width / person.height

        lower_bound = (
            bounds.aspect_ratio()
            - config.thumbnail_person_aspect_ratio_min * bounds.aspect_ratio()
        )
        upper_bound = (
            bounds.aspect_ratio()
            + config.thumbnail_person_aspect_ratio_max * bounds.aspect_ratio()
        )
        # print(f"lowerbounds has aspect_ratio of {lower_bound}")
        # print(f"upperbounds has aspect_ratio of {upper_bound}")
        # print(f"person has aspect_ratio of {aspect_ratio}")

        if lower_bound < aspect_ratio < upper_bound:
            candidate = candidates[random.randrange(0, len(candidates))]
            print(f"selected {candidate}")
            background = Image.new("RGBA", (person.width, person.height), (0, 0, 0, 0))

            # background.paste((0, 0, 0, 255), mask=bordered_person)
            background.paste(person, (0, 0), mask=person)
            # background.show()

            return background

    raise NoSuitablePersonException()


def generate_thumbnail_text_clip(title: str, resolution: Tuple[int, int]) -> TextClip:
    text_max_width = int(resolution[0] * config.thumbnail_text_width_percent)

    text = title.split(" ")
    texts = []
    for index in range(0, len(text), 2):
        if index == len(text) - 1:
            texts.append(text[index])
        else:
            texts.append(text[index] + " " + text[index + 1])

    text = "\n".join(texts).strip()

    txt_clip = TextClip(
        text=text,
        method="caption",
        color=config.text_clips_font_color,
        # font=config.text_clips_font,
        font=config.text_clips_font,
        font_size=100,
        stroke_color=config.text_clips_font_stroke_color,
        stroke_width=6,
        size=(text_max_width, resolution[1]),
        align="west",
    ).with_position((25, 0))

    return txt_clip


def generate_thumbnail_background_clip(background_clip: VideoClip):
    screenshot_time = random.random() * background_clip.duration
    img_clip = ImageClip(background_clip.get_frame(screenshot_time))
    return img_clip


def calculate_person_max_bounds(resolution: Tuple[int, int]):
    allowed_overlap = int(config.thumbnail_allowed_overlap * resolution[0])
    top_left = (
        (config.thumbnail_text_width_percent * resolution[0]) - allowed_overlap,
        0,
    )
    top_right = (resolution[0], 0)
    bottom_left = (
        (config.thumbnail_text_width_percent * resolution[0]) - allowed_overlap,
        resolution[1],
    )
    bottom_right = (resolution[0], resolution[1])

    return Rectangle(top_left, top_right, bottom_left, bottom_right)


def thumbnail_add_person(
    thumbnail: Image.Image, person_max_bounds: Rectangle, person: Image.Image
):
    # scale = min(
    #     person_max_bounds.width() / person.width,
    #     person_max_bounds.height() / person.height,
    # )
    scale = person_max_bounds.height() / person.height
    width = int(person.width * scale)
    height = int(person.height * scale)

    person = person.resize((width, height))

    remaining_width = person_max_bounds.width() - width
    remaining_height = person_max_bounds.height() - height

    thumbnail.paste(
        person,
        (
            int(person_max_bounds.top_left[0]) + int(1 / 2 * remaining_width),
            thumbnail.size[1] - person.size[1],
        ),
        person,
    )

    return thumbnail


def generalize_keywords(keywords: list[str]) -> list[str]:
    change_made = False
    if "mother" in keywords:
        pos = keywords.index("mother")
        keywords[pos] = "woman"
        change_made = True
    if "father" in keywords:
        pos = keywords.index("father")
        keywords[pos] = "man"
        change_made = True

    if not change_made and len(keywords) > 1:
        keywords.remove(keywords[random.randrange(0, len(keywords))])

    return keywords


def generate_thumbnail(
    title: str, background_clip: VideoClip, keywords: list[str], retries_left: int = 1
) -> Image.Image | None:
    try:
        resolution = (1920, 1080)

        background_image_clip = generate_thumbnail_background_clip(background_clip)
        background_image = Image.fromarray(background_image_clip.get_frame(0))

        person_max_bounds = calculate_person_max_bounds(resolution)
        print(
            f"person box dimensions {person_max_bounds.width()}x{person_max_bounds.height()}"
        )

        person: Image.Image = find_thumbnail_person(person_max_bounds, keywords)
        person_and_background = thumbnail_add_person(
            background_image, person_max_bounds, person
        )
        person_and_background_clip: ImageClip = ImageClip(
            np.array(person_and_background)
        ).with_duration(1)

        txt_clip = generate_thumbnail_text_clip(title, resolution)
        clip = CompositeVideoClip(
            [person_and_background_clip, txt_clip], use_bgclip=True
        )
        thumbnail: Image.Image = Image.fromarray(clip.get_frame(0))
        return thumbnail
    # except NoSuitablePersonException as e:
    except Exception as e:
        print(e)
        print(f"Thumbnail generation for keywords {keywords} failed")
        if retries_left > 0:
            new_keywords = generalize_keywords(keywords)
            return generate_thumbnail(
                title, background_clip, new_keywords, retries_left - 1
            )


def generate_thumbnails(post: Post, background_video: VideoClip, video_title: str):
    video_title = video_title[: video_title.index("|")]

    openai_test = OpenAiInterface()

    db_controller = DB_Controller("database.db")
    categories = db_controller.find_all_used_tags(min_amount=10)
    categories.remove("human")
    categories.remove("person")
    categories.remove("happy")
    categories.remove("content")
    categories.remove("man")
    categories.remove("woman")
    print(categories)

    attempts = []
    num_attempts = 10
    thumbnails_remaining = 5

    while True:
        while len(attempts) != num_attempts:
            print("trying to categorize...")
            response = openai_test.generate_text_without_context(
                f"""Categorize this reddit post into these categories: {", ".join(categories)}
Do not use any other categories.
Generate {num_attempts} answers, one per line.
Select 2 categories per answer and dont repeat the same answer.""",
                f"{post.subreddit}\n{post.title}\n{post.selftext}",
            )
            print(response)
            attempts = [attempt.split(", ") for attempt in response.split("\n")]
        print(attempts)

        for attemptNum, categories in enumerate(attempts):
            print(f"Generating Thumbnail {attemptNum}")
            image = generate_thumbnail(video_title, background_video, categories)
            if image != None:
                thumbnails_remaining -= 1
                print("saving")
                image.save(
                    config.output_dir
                    + post.post_id
                    + f"/thumbnail{thumbnails_remaining} - {','.join(categories)}.jpg"
                )

                if thumbnails_remaining == 0:
                    return