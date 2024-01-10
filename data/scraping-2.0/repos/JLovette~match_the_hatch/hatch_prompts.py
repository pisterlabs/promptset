import openai
import json
import requests

from typing import Dict
from collections import defaultdict

PULZE_API_BASE = "https://api.pulze.ai/v1"

EXAMPLE_HATCH_OUTPUT = f"""
Green Drakes (Ephemera guttulata), Green Drake Dry Fly, 10-12, Olive body with gray wings and a touch of brown
Green Drakes (Ephemera guttulata), Green Drake Nymph, 10-12, Olive body with brown and gray accents
Golden Stoneflies, Golden Stonefly Dry, 6-10, Yellow or tan body with dark mottled wings and brown hackle
Golden Stoneflies, Pat's Rubber Legs, 6-10, Dark brown or black body with rubber legs and a touch of orange
Terrestrials, Dave's Hopper, 10-12, Tan or yellow body with a foam wing and rubber legs
Terrestrials, Black Foam Ant, 14-16, Black foam body with a sparse wing and black hackle
"""

EXAMPLE_MATERIALS_OUTPUT = f"""
Elk Hair Caddis, Hook, Size 12-16 dry fly hook
Elk Hair Caddis, Thread, Pink or red 6/0 or 8/0 thread
Elk Hair Caddis, Tail, Medium dun hackle fibers
Elk Hair Caddis, Body, Dubbing in a pinkish-red color
Elk Hair Caddis, Post, White or pink synthetic yarn (for the parachute)
Elk Hair Caddis, Hackle, Brown or grizzly rooster hackle
Elk Hair Caddis, Wing, White or light gray calf body hair
"""

def get_pulze_call(prompt, api_key, labels = {}):
    """
    Generic wrapper for a call to Pulze. 
    """
    print("Calling Pulze...")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Pulze-Labels": json.dumps(labels),
            "Pulze-Weights": json.dumps({
                "cost": 0,
                "quality": 1.0,
                "latency": 0
            })
        }
        payload = {
            "prompt": prompt,  # Pulze doesn't seem to respond to 'prompt' data field: https://docs.pulze.ai/api-reference/chat-completions
            "messages": [{"role": "user", "content": prompt}],
            "best_of": 3,
            "temperature": 1
        }
        response = requests.request("POST", PULZE_API_BASE + "/chat/completions", headers=headers, json=payload).json()
        print(f"Pulze returned {len(response.get('choices'))} options, picking the best one...")
        llm_output = response.get('choices')[0].get('message')
        print(f"Pulze response: {llm_output}")
        return llm_output["content"]
    except Exception as e:
        print(f"Unable to use pulze's chat completion: {repr(e)}")
        return get_openai_call(prompt)

    
def get_openai_call(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    llm_output = response.choices[0].message
    return llm_output["content"]


def generate_hatch_list(location: str, river: str, target_species: str, season: str, api_key) -> Dict[str, Dict]:
    """
    Generate a list of expected hatches and associated fly patterns based on user input
    """
    prompt = get_hatch_prompt(location, river, target_species, season)
    
    content = get_pulze_call(prompt, api_key, labels={"prompt_type": "hatch_list"}).split("\n")
    hatches_to_patterns = llm_hatches_to_dict(content)
    if not len(hatches_to_patterns):
        print("Calling openai instead...")
        content = get_openai_call(prompt).split("\n")
        hatches_to_patterns = llm_hatches_to_dict(content)
    print(hatches_to_patterns)
    return hatches_to_patterns


def llm_hatches_to_dict(content):
    hatches_to_patterns = {}
    for pattern in content:
        try:
            parsed_pattern = pattern.split(', ')
            fly_info = {
                "pattern": parsed_pattern[1],
                "hook_size": parsed_pattern[2],
                "description": parsed_pattern[3],
            }
            if not hatches_to_patterns.get(parsed_pattern[0], None):
                hatches_to_patterns[parsed_pattern[0]] = []
            hatches_to_patterns[parsed_pattern[0]].append(fly_info)
        except Exception as e:
            print(f"Unable to add pattern {pattern} due to unexpected format: {repr(e)}")
    return hatches_to_patterns


def generate_pattern_materials_list(hatches_to_patterns, api_key):
    """
    Generate a shopping list of materials from a previously generated list of recommended fly patterns
    """
    pattern_list_for_materials = "\n"
    for hatch, patterns in hatches_to_patterns.items():
        for pattern in patterns:
            pattern_list_for_materials += hatch + ", " + pattern["pattern"] + ", Size " + pattern["hook_size"] + ", " + pattern["description"] + "\n"
    
    prompt = get_materials_prompt(pattern_list_for_materials)

    llm_output = get_pulze_call(prompt, api_key, labels={"prompt_type": "materials_list"}).split("\n")
    pattern_to_materials = defaultdict(list)
    print(llm_output)
    for line in llm_output:
        try:
            parsed_line = line.split(", ")
            pattern_to_materials[parsed_line[0]].append(([parsed_line[1]], parsed_line[2]))
        except Exception as e:
            print(f"Unable to add material {line} due to unexpected format: {repr(e)}")

    return pattern_to_materials


def get_hatch_prompt(location: str, river: str, target_species: str, season: str):
    return (
        f"""
        I am planning a fly-fishing trip to {location}, where I will be targeting {target_species} on the {river}. 
        The trip will be in {season}. Predict at least 5 of the insect hatches that will be going on in this area at this point in the season, 
        and suggest at least two patterns for each of the insect species. Include the hook sizes 
        and colors of the recommended fly patterns. Return each fly pattern on a new line in the following format:
        
        Insect Species (optional latin name), Fly Pattern Name, Hook Size, Color Description

        Example Output:
        {EXAMPLE_HATCH_OUTPUT}
        """
    )


def get_materials_prompt(pattern_list_for_materials):
    return (
        f"""
        Generate and combine a complete shopping list of materials for a list of fly fishing patterns. Do not include any other headers or information, only the cumulative list of recommended materials.
        Format each line of the output in the following format:

        Pattern, Component, Description

        An example input list and desired output is given below:

        Example fly pattern: 
        Caddisflies, Elk Hair Caddis, Size 14-18, Light tan or brown body with elk hair wings and a brown hackle

        Example output:
        {EXAMPLE_MATERIALS_OUTPUT}

        Generate the material shopping list for the following list of patterns:
        {pattern_list_for_materials}
        """
    )