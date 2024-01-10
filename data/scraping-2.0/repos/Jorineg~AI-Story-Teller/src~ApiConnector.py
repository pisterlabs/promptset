import yaml
from openai import OpenAI, RateLimitError
from elevenlabs import generate, save, set_api_key, Voice, VoiceSettings
import os
import requests
import base64
from JsonParser import parse_incomplete_json, parse_json
from JsonSchemas import (
    validate_object,
    character_details_schema,
    baseline_schema,
    story_image_schema,
)
import json
import copy
import time
from config import (
    ROOT_PATH,
    MAX_RETRY_RESPONSE_FORMAT_FAIL,
    prompt_params,
    MAX_IMAGE_RETRY_COUNT,
)
import logging
from Logging import create_gpt_log, add_gpt_log_response, complete_gpt_log
import re

logger = logging.getLogger(__name__)

# dotenv loaded in mainpage.py
# openai loads api key from environment variable automatically
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

default_model_parameters = {
    "model": "gpt-3.5-turbo",
    "temperature": 1,
    "max_tokens": 100,
}

stored_prompts = {}
with open(f"{ROOT_PATH}/prompts.yaml", "r") as f:
    stored_prompts = yaml.safe_load(f)


client = OpenAI()


def gpt_request(
    prompt_name, model_input, use_gpt_3_turbo=False, extra_prompt_params={}
):
    prompt = stored_prompts[prompt_name]
    all_prompt_params = {**prompt_params, **extra_prompt_params}
    for key, value in all_prompt_params.items():
        prompt["prompt"] = prompt["prompt"].replace(f"{{{key}}}", str(value))
    params = default_model_parameters.copy()
    params.update(prompt["parameters"])

    if use_gpt_3_turbo:
        params["model"] = "gpt-3.5-turbo"

    messages = [
        {"role": "system", "content": prompt["prompt"]},
        {"role": "user", "content": model_input},
    ]

    logger.info(f"attempting to create chat completion for {prompt_name}")
    try:
        response = client.chat.completions.create(
            messages=messages,
            stream=True,
            **params,
        )
    except RateLimitError as e:
        # wait for 1 second and retry
        logger.warning("rate limit error, retrying")
        time.sleep(1)
        return gpt_request(prompt_name, model_input)
    logger.info(f"successfully created chat completion for {prompt_name}")

    gpt_log_identifier = create_gpt_log(
        prompt_name, prompt["prompt"], model_input, params
    )
    return gpt_request_yields(response, gpt_log_identifier, prompt_name)


def gpt_request_yields(openai_response, gpt_log_identifier, prompt_name):
    combined_response = ""
    for chunk in openai_response:
        if chunk.choices[0].finish_reason not in ["stop", None]:
            logger.error(
                f"Chat gpt returned an error\nFinish reason: {chunk.choices[0].finish_reason}\nPrompt name: {prompt_name}"
            )
            if combined_response == "":
                raise RuntimeError(
                    "Chat gpt returned an error\n\nFinish reason: "
                    + chunk.choices[0].finish_reason
                    + "\n\nPrompt name:\n"
                    + prompt_name
                )
            else:
                logger.warning(
                    f"ignoring error and continuing with partial response for {prompt_name}"
                )
        if chunk.choices[0].delta.content is not None:
            chunk_content = chunk.choices[0].delta.content
            combined_response += chunk_content
            add_gpt_log_response(gpt_log_identifier, chunk_content)
            yield chunk_content
    logger.info(f"finished chat completion for {prompt_name}")
    complete_gpt_log(gpt_log_identifier)


def validate_query(query):
    response = "".join(gpt_request("validate_prompt", query))
    return response


def generate_story_ideas(_, query):
    response = "".join(gpt_request("generate_story_ideas", query))
    return response


def choose_story_idea(story_ideas, query, retry=0):
    if retry > MAX_RETRY_RESPONSE_FORMAT_FAIL:
        raise Exception("Failed to choose story idea")
    model_input = f"Prompt/Title: {query}\n\nStory ideas:\n{story_ideas}"
    response = "".join(gpt_request("choose_story_idea", model_input))
    chosen_idea = response.split("BEST BASELINE:")
    if len(chosen_idea) < 2:
        return choose_story_idea(story_ideas, query, retry + 1)
    return chosen_idea[1].strip()


def generate_story_json(incremental, query, summary_idea):
    response = ""
    parsed_character_infos = []
    yielded_character_infos = set()
    model_input = f"""
        Story prompt/title: {query}

        Summary idea: {summary_idea}
    """
    for chunk in gpt_request("generate_story_json", model_input):
        response += chunk
        parsed_json = parse_incomplete_json(response)

        if (
            parsed_json
            and "characters" in parsed_json
            and len(parsed_json["characters"]) - 1 > len(parsed_character_infos)
        ):
            parsed_character_infos = parsed_json["characters"][:-1]
            for parsed_character_info in parsed_character_infos:
                if parsed_character_info["name"] not in yielded_character_infos:
                    incremental(
                        parsed_character_info, parsed_json["place"], parsed_json["time"]
                    )
                    yielded_character_infos.add(parsed_character_info["name"])

    parsed_character_infos = parse_incomplete_json(response)["characters"]
    incremental(parsed_character_infos[-1], parsed_json["place"], parsed_json["time"])

    response_object = parse_json(response)
    validate_object(response_object, baseline_schema)
    return response_object


def generate_paragraph_headings(story_json, story_summary):
    modified_story_json = {}
    important_information = [
        "title",
        "language",
        "place",
        "time",
        "characters",
        "objects",
    ]
    for key in important_information:
        modified_story_json[key] = story_json[key]
    modified_story_json["summary"] = story_summary
    modified_story_json = json.dumps(modified_story_json)

    response = "".join(gpt_request("generate_paragraph_headings", modified_story_json))
    return response


# character details in the order of the characters in the story json
def generate_next_paragraph(
    paragraph_nr,
    story_json,
    paragraph_headings,
    previous_paragraphs,
    *character_details,
):
    modified_story_json = {}
    important_information = [
        "title",
        "language",
        "place",
        "time",
        "objects",
        "narrator",
    ]
    for key in important_information:
        modified_story_json[key] = story_json[key]
    modified_story_json["characters"] = character_details
    modified_story_json = json.dumps(modified_story_json)

    line_break = "\n"
    double_line_break = "\n\n"

    if previous_paragraphs:
        list_previous_paragraphs = [
            f"P{i+1}:{line_break+p}" for i, p in enumerate(previous_paragraphs)
        ]

    model_input = f"""
        Simulation JSON:
        {modified_story_json}

        Outline:
        {paragraph_headings}

        Previous Paragraphs:
        {f"{double_line_break.join(list_previous_paragraphs)}" if previous_paragraphs else "None"}

        P{paragraph_nr}:
    """

    # use_gpt_3_turbo = paragraph_nr > 3
    use_gpt_3_turbo = False

    extra_prompt_params = {"PARAGRAPH_NUMBER": paragraph_nr}
    response = "".join(
        gpt_request(
            "generate_next_paragraph",
            model_input,
            use_gpt_3_turbo=use_gpt_3_turbo,
            extra_prompt_params=extra_prompt_params,
        )
    )
    # check if response contains "PX:" (and optional new line) where X is any number
    # first check for match at very beginning of response
    # if it exists, replace with nothing
    # if there is another match, split the response at that match and return the first part
    match = re.match(r"^(P\d+:?\n?)", response)
    if match:
        response = response.replace(match.group(1), "", 1)
        logger.info(
            f"response contains leading paragraph number in paragraph {paragraph_nr}"
        )
    match = re.match(r"(P\d+:?\n?)", response)
    if match:
        response = response.split(match.group(1))[0]
        logger.warning(f"response format failed for paragraph {paragraph_nr}")
    return response


def generate_image_json(
    fist_paragraph, previous_paragraph, paragraph, story_json, retry=0
):
    modified_story_json = {}
    important_information = [
        "place",
        "time",
        "characters",
        "objects",
    ]
    for key in important_information:
        modified_story_json[key] = story_json[key]

    modified_story_json = json.dumps(modified_story_json)

    model_input = f"""
        Story JSON:
        {modified_story_json}

        First Paragraph:
        {fist_paragraph}

        Previous Paragraph:
        {previous_paragraph if previous_paragraph else "None"}

        Current Paragraph (generate image idea for this paragraph):
        {paragraph}
    """

    response = "".join(gpt_request("generate_image_json", model_input))

    def retry(error_message):
        if retry > MAX_RETRY_RESPONSE_FORMAT_FAIL:
            raise Exception(error_message)
        logger.warning(error_message)
        return generate_image_json(
            fist_paragraph, previous_paragraph, paragraph, story_json, retry + 1
        )

    response_object = parse_json(response, throw_error=False)
    if response_object is None:
        return retry("Failed to generate image json. Could not parse json")

    is_valid = validate_object(
        response_object, story_image_schema, raise_exception=False
    )
    if not is_valid:
        return retry("Failed to generate image json. Output format is not valid")

    return response_object


def generate_character_details(story_summary, character_infos, place, time, retry=0):
    character_infos_str = json.dumps(character_infos)
    model_input = f"""
        Summary:
        {story_summary}

        Place:
        {place}

        Time:
        {time}

        Character:
        {character_infos_str}
    """
    response = "".join(gpt_request("generate_character_details", model_input))
    try:
        response_object = parse_json(response)
        validate_object(response_object, character_details_schema)
    except:
        if retry > MAX_RETRY_RESPONSE_FORMAT_FAIL:
            raise Exception(
                "Failed to generate character details. Output format is not valid"
            )
        return generate_character_details(story_summary, character_infos, retry + 1)

    result_character_infos = copy.deepcopy(character_infos)
    response_object = {**response_object, **result_character_infos}
    return response_object


def generate_stable_diffusion_prompt(image_json, story_json, *character_details):
    charcter_map = {details["name"]: details for details in character_details}
    object_map = {obj["name"]: obj for obj in story_json["objects"]}

    def get_character_details(name):
        if name in charcter_map:
            return charcter_map[name]
        logger.warning(f"Could not find character details for {name}")
        return {
            "name": name,
            "error": "no character details found. Please make up some details for this character.",
        }

    def get_object_details(name):
        if name in object_map:
            return object_map[name]
        logger.warning(f"Could not find object details for {name}")
        return {
            "name": name,
            "error": "no object details found. Please make up some details for this object.",
        }

    image_character_details = [
        get_character_details(name) for name in image_json["characters"]
    ]
    image_object_details = [get_object_details(name) for name in image_json["objects"]]
    image_json["characters"] = image_character_details
    image_json["objects"] = image_object_details

    broad_image_style_map = {
        "photography": ["analog-film", "photographic"],
        "painting": ["isometric", "line-art"],
        "digital painting": ["fantasy-art", "cinematic"],
        "digital art": [
            "digital-art",
            "modeling-compound",
            "origami",
            "tile-texture",
            "3d-model",
            "low-poly",
            "neon-punk",
            "enhance",
        ],
        "digital illustration": ["pixel-art", "3d-model"],
        "illustration": ["anime", "comic-book", "pixel-art"],
    }

    for key, value in broad_image_style_map.items():
        if story_json["image_style"] in value:
            story_json["image_style"] = key
            break

    image_json = json.dumps(image_json)

    response = "".join(gpt_request("generate_stable_diffusion_prompt", image_json))
    return response


def generate_audio(narrator, section_nr, text, story_id):
    logger.debug(f"generating audio for section {section_nr} with narrator {narrator}")
    narrator_map = {
        "Vanessa": "3ARdf6wTw5HVbUm0EVtt",
        "Josh": "BAQmo5xF1wQVvtrEmoNY",
        "Bella": "9RSRUmC0WRdWpNCmcO92",
        "Matthew": "eqFDBBoxsDMCl1xXmRhZ",
        "Ann": "GF9c4URMiVpQ372hFkjL",
        "Winston": "SmJU8f86ds5LWKhsyDw3",
    }
    backup_voice = narrator_map["Matthew"]
    narrator_id = narrator_map.get(narrator, backup_voice)

    audio = generate(
        text=text,
        # Ryan Kurk?
        # "old wise man"
        voice=Voice(
            voice_id=narrator_id,
            settings=VoiceSettings(
                stability=0.2, similarity_boost=0.5, style=0.0, use_speaker_boost=False
            ),
        ),
        model="eleven_multilingual_v2",
    )
    save(audio, f"{ROOT_PATH}/stories/{story_id}/sounds/{section_nr}.mp3")
    return True


def generate_image(section_nr, image_prompt, story_id, baseline, retry=0):
    api_host = "https://api.stability.ai"
    engine_id = "stable-diffusion-xl-1024-v1-0"
    generation_endpoint = f"/v1/generation/{engine_id}/text-to-image"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}",
    }

    negative_prompt = """
    worst quality, normal quality, low quality, low res,
    blurry, text, watermark, logo, banner, extra digits,
    cropped, jpeg artifacts, signature, username, error,
    sketch ,duplicate, ugly, monochrome, geometry,
    mutation, disgusting, duplicate character, too many fingers,
    nude, nsfw, mutated body parts, disfigured, bad anatomy,
    deformed body features 
    """

    json = {
        "text_prompts": [
            {
                "text": image_prompt,
                "weight": 1,
            },
            {
                "text": negative_prompt,
                "weight": -1,
            },
        ],
        "cfg_scale": 7,
        "height": 768,
        "width": 1344,
        "samples": 1,
        "steps": 40,
        "style_preset": baseline["image_style"],
    }
    response = requests.post(
        api_host + generation_endpoint,
        headers=headers,
        json=json,
    )
    if response.status_code != 200:
        if retry > MAX_IMAGE_RETRY_COUNT:
            raise Exception("Image generation failed\n" + response.text)
        logger.warning(
            f"Image generation failed for section {section_nr}. Reason: {response.text}"
        )
        time.sleep(1)
        return generate_image(section_nr, image_prompt, story_id, baseline, retry + 1)
    data = response.json()
    image = data["artifacts"][0]
    if image["finishReason"] != "SUCCESS":
        if retry > MAX_IMAGE_RETRY_COUNT:
            raise Exception("Image generation failed\n" + image["finishReason"])
        logger.warning(
            f"Image generation failed for section {section_nr}. Reason: {image['finishReason']}"
        )
        return generate_image(section_nr, image_prompt, story_id, baseline, retry + 1)
    image_data = base64.b64decode(image["base64"])
    with open(f"{ROOT_PATH}/stories/{story_id}/images/{section_nr}.png", "wb") as f:
        f.write(image_data)
    return True
