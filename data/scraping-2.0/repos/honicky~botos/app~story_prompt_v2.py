
import os
import openai
import json
import random
import re


from story_config import story_config
from util import current_time_ms, url_to_pil_image
import image_prompt
import llm_prompt
import object_store_client
import mongo_client
import prompt_templates

import wandb
from wandb.sdk.data_types.trace_tree import Trace

run = wandb.init(project="story-time")

root_span = Trace(
    name="StoryGenAgent",
    kind="agent",
    start_time_ms= current_time_ms(),
    metadata={"version": "0.2"},
)

def generate_story_outline():
    start_time = current_time_ms()
    story_outline = {
        "StoryStructure": random.choice(story_config["StoryStructure"]),
        "Setting": random.choice(story_config["Setting"]),
        "Character": random.choice(story_config["Characters"]),
        "Style": random.choice(story_config["Style"]),
        "Tone": random.choice(story_config['Tone']),
        "CulturalSetting": "None",
        "EverydayPublicPlace": "None"
    }
  
    # # 25% chance of having a cultural setting
    # if random.random() < 0.25:
    #     story_outline["CulturalSetting"] = random.choice(story_config["CulturalSettings"])
    # else:
    #     story_outline["CulturalSetting"] = "None"

    # If the setting is an EverydayPublicPlace
    if story_outline["Setting"]["type"] == "Everyday Public Place":
        story_outline["EverydayPublicPlace"] = random.choice(story_config["EverydayPublicPlaces"])

    end_time = current_time_ms()
    story_outline_span = Trace(
        name="GenerateStoryOutline",
        kind="tool",
        status_code="success",
        outputs=story_outline,
        start_time_ms=start_time,
        end_time_ms=end_time,
    )
    root_span.add_child(story_outline_span)
    root_span.add_inputs_and_outputs(
        inputs={}, outputs=story_outline
    )
    root_span._span.end_time_ms = end_time

    return story_outline

def generate_setting_prompt(story_outline):
    selected_setting = story_outline["Setting"]
    specific_setting = story_outline["EverydayPublicPlace"]
  
    return prompt_templates.setting_prompt(
        selected_setting["type"],
        selected_setting["description"],
        selected_setting["example"],
        story_outline["Tone"]["type"],
        story_outline["Tone"]["description"],
        story_outline["Tone"]["example"],
        specific_setting["type"] if specific_setting != "None" else None,
        specific_setting["example"] if specific_setting != "None" else None,
    )

def extract_paragraphs_from_story(story_text):
    start_time = current_time_ms()

    paragraphs = [
        paragraph.strip()
        for paragraph in story_text.split("\n")
        if paragraph.strip() != ""
    ]

    paragraph_extractor_span = Trace(
        name="ExtractParagraphsFromStory",
        kind="tool",
        status_code="success",
        outputs={"paragraphs": paragraphs},
        start_time_ms=start_time,
        end_time_ms=current_time_ms(),
    )
    root_span.add_child(paragraph_extractor_span)
    root_span.add_inputs_and_outputs(
        inputs={"story_text": story_text}, outputs={"paragraphs": paragraphs}
    )
    root_span._span.end_time_ms = paragraph_extractor_span._span.end_time_ms
    return paragraphs

def store_story_in_s3(story_descriptor, run_name):
    bucket_id = "botos-generated-images"
    user_id = "dev-test-user"
    story_descriptor_key = f"{user_id}/story_prompt_v2/{run_name}.json"
    client = object_store_client.Boto3Client()
    client.upload_object(story_descriptor, bucket_id, story_descriptor_key)
    return story_descriptor_key

def store_story_in_mongo(story_descriptor):
    mongo = mongo_client.StoryMongoDB()
    story_id = None
    try:
        story_id = mongo.insert_story(story_descriptor)
    finally:
        mongo.close_connection()
    return story_id

try:
    # next_leg_token = os.environ.get("THE_NEXT_LEG_API_TOKEN")
    # image_client = image_prompt.NextLegClient(next_leg_token)
    # dalle_client = DalleClient(openai_api_token)
    # beam_api_token = os.environ.get("BEAM_SECRET_KEY_UUENCODED")
    # image_client = image_prompt.StableDiffusionClient(beam_api_token, "botos-generated-images", "dev-test-user")
    image_client = image_prompt.ReplicateClient("sdxl")

    def generate_image(prompt):
        print(f"Generating image for prompt: {prompt} ...", end="", flush=True )
        start_time = current_time_ms()
        try:
            image_urls = image_client.generate_image(prompt)
            status = "success"
            # return next_leg_client.generate_image(prompt),
            print(f"Done")
            print(f"Image: {image_urls}")

        except Exception as e:
            print(f"Error generating image: {e}")
            status = "error"
            image_urls = []

        generate_image_span = Trace(
            name="GenerateImage",
            kind="tool",
            status_code=status,
            inputs={"prompt": prompt},
            outputs={
                "url": image_urls,
                # "images": wandb.Image(url_to_pil_image(image_urls["url"]), caption=prompt)
            },
            start_time_ms=start_time,
            end_time_ms=current_time_ms(),
        )
        root_span.add_child(generate_image_span)
        root_span.add_inputs_and_outputs(
            inputs={"prompt": prompt}, outputs={"image_ulrs": image_urls}
        )
        root_span._span.end_time_ms = generate_image_span._span.end_time_ms

        return image_urls
    generate_images = True

    # Generate the story outline first
    story_outline = generate_story_outline()
    story = {
        "outline": story_outline
    }
    # Then generate the setting prompt based on the outline
    setting_prompt = generate_setting_prompt(story_outline)

    print("Generating story setting... ", end="", flush=True)

    openai_api_token = os.environ.get("OPENAI_API_TOKEN")
    llm = llm_prompt.GTP4LLM(openai_api_token)

    # Get the story setting from GPT-4
    story_setting= llm.generate_text(setting_prompt, root_span)
    story["setting"] = {
        "setting_text": story_setting,    
    }
    # Print the generated story setting
    print(f"Done")

    character_prompt_str = prompt_templates.generate_character_prompt_for_gpt(
        story_outline["Character"]["type"],
        story_outline["Character"]["description"],
        story_outline["Character"]["example"]
    )

    print("Generating the main character... ", end="", flush=True)
    character_description = llm.generate_text(character_prompt_str, root_span)
    print(f"Done")

    story["character"]= {
        "description": character_description,
    }

    story_prompt = prompt_templates.generate_story_prompt(story)

    llm.reset_history()
    print("Generating the story... ", end="", flush=True)
    story["story_text"] = llm.generate_text(story_prompt, root_span)
    print(f"Done")

    llm.reset_history()
    print("Generating the setting image prompt... ", end="", flush=True)
    story["setting"]["image_prompt"] = llm.generate_text(prompt_templates.generate_setting_prompt_for_sdxl(story["setting"]["setting_text"]), root_span, model_name="gpt-3.5-turbo")
    print(f"Done")

    llm.reset_history()
    print("Generating the character image prompt... ", end="", flush=True)
    story["character"]["image_prompt"] = llm.generate_text(prompt_templates.generate_setting_prompt_for_sdxl(story["character"]["description"]), root_span, model_name="gpt-3.5-turbo")
    print(f"Done")

    paragraphs = extract_paragraphs_from_story(story["story_text"])

    # print(f"paragraphs: {paragraphs}")
    # print("Generating the story image prompt... ", end="", flush=True)

    if generate_images:
        paragraph_image_prompts = [
            llm.generate_text(
                prompt_templates.generate_paragraph_image_prompt(
                    story["setting"]["setting_text"],
                    story["character"]["description"],
                    paragraph
                ),
                root_span,
            ) if llm.reset_history() is None else ""
            for paragraph in paragraphs
        ]
        image_prompts = [
            # SDXL Lora prompt
            # f"({paragraph_image_prompt}:2) ({story['character']['image_prompt']}:1.5) {story['setting']['image_prompt']}"

            # Midjourney artist prompt
            # f"Seymour Chwast's depiction of {paragraph_image_prompt} {story['character']['image_prompt']} {story['setting']['image_prompt']}"
            # f"Arthur Dove's painting depicting {paragraph_image_prompt} {story['character']['image_prompt']} {story['setting']['image_prompt']}"
            f"David Burliuk's painting depicting {paragraph_image_prompt} {story['character']['image_prompt']} {story['setting']['image_prompt']}"
            #"Maud Lewis's painting depicting"
            # Midjourney style prompt
            # f"{paragraph_image_prompt} {story['character']['image_prompt']} {story['setting']['image_prompt']} --style 2cz4gHl6qBa7MiJ6"
            # f"{paragraph_image_prompt} {story['character']['image_prompt']} {story['setting']['image_prompt']} in the style of Watercolor Paint"

            for paragraph_image_prompt in paragraph_image_prompts
        ]
        story["pages"] = [
            {
                "paragraph": paragraph,
                "paragraph_image_prompt": paragraph_image_prompt,
                "image_prompt": image_prompt,
                "image_urls": generate_image(image_prompt) 
            }
            for paragraph, paragraph_image_prompt, image_prompt in zip(paragraphs, paragraph_image_prompts, image_prompts)
        ]

    story["run_name"] = run.name
    story["version"] = "v2.1.0"
    story_id = store_story_in_mongo(story)

    print(f"Story id: {story_id}")

finally:
    root_span._span.end_time_ms = current_time_ms()
    root_span.log(name="story-time")
