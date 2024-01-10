import openai
import os
import json
from dotenv import load_dotenv
from textwrap import dedent
from typing import List, Dict
from helpers.path_helper import GENERATED_PROMPS_DIR, PANELS_TEXT_DIR
from helpers.aws_helper import (
    S3_BUCKET,
    load_json_from_s3,
    save_json_to_s3,
    get_s3_connection,
)

GPT_MODEL = "gpt-3.5-turbo"

load_dotenv("./api_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt_demo():
    demo_description = [
        {
            "characters": ["batman", "superman"],
            "visual_context": "on a rooof, city in background, nightsky",
            "text": ["I'm the boss", "No I am !"],
        },
        {
            "characters": ["batman", "superman"],
            "visual_context": "on the streets",
            "text": ["You dead bro"],
        },
    ]

    generated_prompts = ask_gpt(demo_description)

    print(generated_prompts)
    output_path = f"{GENERATED_PROMPS_DIR}/promps.json"
    with open(output_path, "w") as outfile:
        json.dump(generated_prompts, outfile)


def generate_prompts():
    previous_descriptions = get_previous_descriptions()
    generated_prompts = ask_gpt(previous_descriptions)
    save_json_to_s3(generated_prompts, GENERATED_PROMPS_DIR)


def ask_gpt(
    previous_panels_description: List[Dict],
    nb_panels_to_generate: int = 4,
) -> List[Dict]:
    messages = set_message(nb_panels_to_generate, previous_panels_description)
    response = openai.ChatCompletion.create(model=GPT_MODEL, messages=messages)
    new_prompts = extract_panels_prompts(response)
    return new_prompts


def extract_panels_prompts(response: Dict) -> List[Dict]:
    prompts_str = response["choices"][0]["message"]["content"]
    prompts_list = split_prompts_str(prompts_str)
    return prompts_list


def split_prompts_str(prompts_str) -> List[Dict]:
    prompts = prompts_str.split("\n\n")
    result = []
    for prompt in prompts:
        tmp = prompt.split("\n")[1:]
        dict = {}
        dict["prompt"] = f"{tmp[0].split(':')[1]} {tmp[1].split(':')[1]}"
        dict["text"] = tmp[2].split(":")[1]
        result.append(dict)

    return result


def format_panels_description(previous_panels_description: List[Dict]):
    result = ""

    for i, panel in enumerate(previous_panels_description):
        result += dedent(
            f"""
            panel {i+1}:
            characters: {', '.join(panel['characters'])}
            visual_context: {panel["visual_context"]}
            text: {', '.join(panel["text"])}"""
        )

    return result


def set_message(
    nb_panels_to_generate: int,
    previous_panels_description: List[Dict],
):
    return [
        {
            "role": "system",
            "content": dedent(
                """
                    You are a comics writer, 
                    when you write a panel you have to describe it as following: 
                    give principals characters, the action performed, and visual context. 
                    A panel need to be a single sentences.
                    exemple: batman talking to spiderman on a roof, nightsky, city in background"""
            ),
        },
        {
            "role": "user",
            "content": dedent(
                f"""
                    Here are a description of a comics page, panels by panels:
                    {format_panels_description(previous_panels_description)}
                    Write {nb_panels_to_generate} panels that follow this story."""
            ),
        },
    ]


def get_previous_descriptions():
    descriptions = []
    previous_panels_text = load_json_from_s3(PANELS_TEXT_DIR)
    for panel_text_file in previous_panels_text:
        current_description = {
            "characters": [
                "batman",
                "superman",
            ],  # because no probant result with computer vision
            "visual_context": "night, city",  # because no probant result with computer vision
            "text": [],
        }

        # get_panels_text(panel_text_file, current_description)
        current_description["text"].append(panel_text_file["Text"])
        descriptions.append(current_description)

    return descriptions
