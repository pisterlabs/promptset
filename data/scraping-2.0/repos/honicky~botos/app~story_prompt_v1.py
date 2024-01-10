
import os
import openai
import json
import random
import re

from story_config import story_config
import image_prompt
import llm_prompt
import prompt_templates

def generate_story_outline():
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

def generate_character_prompt(story_outline):
    character_type = story_outline['Character']
    prompt = """Please provide a detailed character description for the following character type:
{char_type}

Feel free to include their personality, appearance, background, or any other relevant details."""
    return prompt


def extract_setting_prompts(input_str):
    # Initialize the dictionary to store the results
    result = {'cover_setting': None, 'setting_prompts': []}
    
    # Use regex to find the content within <cover-setting> tags
    cover_setting_match = re.search(r'<cover-setting>(.*?)<\/cover-setting>', input_str, re.DOTALL)
    if cover_setting_match:
        result['cover_setting'] = cover_setting_match.group(1).strip()
    
    # Use regex to find the content within multiple <setting> tags
    setting_matches = re.findall(r'<setting>(.*?)<\/setting>', input_str, re.DOTALL)
    if setting_matches:
        # Clean up the extracted prompts by removing extra whitespace
        result['setting_prompts'] = [setting.strip() for setting in setting_matches]
        
    return result

def extract_character_descriptions(response_text):
    pattern = r'<character-description>(.*?)<\/character-description>'
    descriptions = re.findall(pattern, response_text, re.DOTALL)

    return descriptions


def parse_numbered_list(input_str, max_lines=3):
    # Regular expression to match numbered list items
    # This pattern is relatively insensitive to formatting and whitespace
    # and captures multiple lines up to the specified max_lines
    pattern = r'\s*\d+\.\s*((?:.*?(?:\n|$)){1,' + str(max_lines) + '})'

    # Find all matches in the input string
    list_items = re.findall(pattern, input_str)

    # Clean up the captured items
    cleaned_items = [' '.join(item.strip().split('\n')) for item in list_items]

    # Return the list of items
    return cleaned_items

generate_images = False

# Generate the story outline first
story_outline = generate_story_outline()
story = {
    "outline": story_outline
}
# Then generate the setting prompt based on the outline
setting_prompt = generate_setting_prompt(story_outline)

print(f"\nGenerated Setting Prompt: {setting_prompt}")

openai_api_token = os.environ.get("OPENAI_API_TOKEN")
llm = llm_prompt.GTP4LLM(openai_api_token)

# Get the story setting from GPT-4
story_setting= llm.generate_text(setting_prompt)
story["setting"] = {
    "setting_text": story_setting,    
}
# Print the generated story setting
print(f"\nGenerated Story Setting: {story['setting']['setting_text']}")

detailed_setting_prompt = prompt_templates.generate_detailed_setting_prompt()

setting_str = llm.generate_text(detailed_setting_prompt)
setting_prompts = extract_setting_prompts(setting_str)
story["setting"].update(setting_prompts)

# next_leg_token = os.environ.get("NEXT_LEG_TOKEN")
# next_leg_client = NextLegClient(next_leg_token)
# dalle_client = DalleClient(openai_api_token)
beam_api_token = os.environ.get("BEAM_SECRET_KEY_UUENCODED")
image_client = image_prompt.StableDiffusionClient(beam_api_token, "botos-generated-images", "dev-test-user")

def generate_image(prompt):
    try:
        return image_client.generate_image(prompt)
        # return next_leg_client.generate_image(prompt),
    except Exception as e:
        print(f"Error generating image: {e}")
        return []

if generate_images:
    story["setting"]["images"] = {
        "cover-image": generate_image(setting_prompts["cover_setting"]),
        "setting-images": [
            generate_image(setting_prompt)
            for setting_prompt in setting_prompts["setting_prompts"]
        ]
    }
    print (f"\nGenerated Setting Images: {json.dumps(story['setting']['images'], indent=2)}")

else:
    print(story["setting"]["setting_prompts"])

character_prompt_str = prompt_templates.generate_character_prompt_for_gpt(
    story_outline["Character"]["type"],
    story_outline["Character"]["description"],
    story_outline["Character"]["example"]
)

character_description = llm.generate_text(character_prompt_str)
print(f"\nGenerated Character: {character_description}")

story["character"]= {
    "description": character_description,
}

# Generate the character prompts
character_image_prompt_generation_prompt = prompt_templates.generate_character_image_prompt()
character_image_prompt_str = llm.generate_text(character_image_prompt_generation_prompt)
# print(f"\nGenerated Character Prompts: {character_midjourney_prompt_str}")

story["characters"]["prompts"] = extract_character_descriptions(character_image_prompt_str)

# Print the generated story setting
print(f"\nGenerated Character Prompts: {story['characters']['prompts']}")

if generate_images:
    story["characters"]["images"] = [
        generate_image(f"""{character_prompt}. blank white background.""")
        for character_prompt in story['characters']['prompts']
    ]

    print(f"\nGenerated Character Images: {json.dumps(story['characters']['images'], indent=2)}")
else:
    print(story["characters"]["prompts"])

story_skeleton_prompt = prompt_templates.generate_story_skeleton_prompt(story)
print(f"\nStory Skeleton Prompt: {story_skeleton_prompt}")

llm.reset_history()
story_skeleton = llm.generate_text(story_skeleton_prompt)
story["story_skeleton"] = story_skeleton

print(f"\nGenerated Story Skeleton: {story['story_skeleton']}")

story["scenes"] = {
    "skeleton": parse_numbered_list(story["story_skeleton"])
}

# llm.reset_history()
# story_image_prompts = prompt_templates.generate_scene_image_prompt(story)
# synthesized_images_prompt = prompt_templates.generate_scene_image_prompt(story)

# print(f"\nSynthesized Images Prompt: {synthesized_images_prompt}")
# synthesized_images_prompts_str = llm.generate_text(synthesized_images_prompt)

# print(f"\nGenerated Synthesized Images Prompt: {synthesized_images_prompts_str}")

# parsed_story_skeleton = llm.generate_text(
#     prompt_templates.parse_story_skeleton_prompt()
# )
# print(f"\nParsed story skeleton: {parsed_story_skeleton}")

# if generate_images:
if False:
    story["scenes"]["images"] = [
        generate_image(scene)
        for scene in story['scenes']['skeleton']
    ]

    print(f"\nScenes: {json.dumps(story['scenes'], indent=2)}")
else:
    print(story["scenes"]["skeleton"])
