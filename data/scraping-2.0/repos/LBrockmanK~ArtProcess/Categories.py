import openai
import json
from tqdm import tqdm
import os

def analyze_tags(tag):
    # Generate LLM tag descriptions

    llm_prompts = [ f"Describe concisely what a(n) {tag} looks like:", \
                    f"How can you identify a(n) {tag} concisely?", \
                    f"What does a(n) {tag} look like concisely?",\
                    f"What are the identifying characteristics of a(n) {tag}:", \
                    f"Please provide a concise description of the visual characteristics of {tag}:"]

    results = {}
    result_lines = []

    result_lines.append(f"a photo of a {tag}.")

    for llm_prompt in tqdm(llm_prompts):

        # send message
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "assistant", "content": llm_prompt}],
            max_tokens=77,
            temperature=0.99,
            n=10,
            stop=None
        )

        # parse the response
        for item in response.choices:
            result_lines.append(item.message['content'].strip())
        results[tag] = result_lines
    return results

categories =[
'Creature/Humanoid/Human',
'Creature/Humanoid/Elf',
'Creature/Humanoid/Bestial',
'Creature/Humanoid/Goblin',
'Creature/Humanoid/Merfolk',
'Creature/Humanoid/Orc',
'Creature/Aquatic',
'Creature/Arthropod',
'Creature/Avian',
'Creature/Reptilian',
'Creature/Amphibian',
'Creature/Amorphous',
'Creature/Robotic',
'Creature/Undead',
'Creature/Elemental',
'Creature/Corrupted',
'Size/Small',
'Size/Medium',
'Size/Large',
'Sex/Male',
'Sex/Female',
'Sex/Androgynous',
'Genre/Fantasy',
'Genre/Modern',
'Genre/Sci-fi',
'Genre/Horror',
'Genre/Historical',
'Genre/Western',
'Things/Vehicles',
'Things/Equipment',
'Things/Food',
'Things/Items',
'Things/Landscapes',
'Composition/Group',
'Composition/Solo',
'Composition/Action',
'Composition/Portrait',
'Composition/Landscape',
'Mood/Serious',
'Mood/Humorous',
'Mood/Dark',
'Mood/Light-hearted',
'Environment/Urban',
'Environment/Rural',
'Environment/Dungeon',
'Environment/Space',
'Environment/Forest',
'Environment/Mountain',
'Style/Sketch',
'Style/Digital',
'Style/Watercolor',
'Style/Photo',
'Other/HasText',
'Other/Meme'
]

if __name__ == "__main__":
    # set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')  # Make sure the environment variable is set

    tag_descriptions = []

    for tag in categories:
        result = analyze_tags(tag)
        tag_descriptions.append(result)

    output_file_path = 'Categories.json'

    with open(output_file_path, 'w') as w:
        json.dump(tag_descriptions, w, indent=3)
