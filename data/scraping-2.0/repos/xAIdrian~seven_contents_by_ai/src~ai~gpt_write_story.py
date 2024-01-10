import sys
import os
sys.path.append("../src")

import openai
from time import time,sleep
import utility.utils as utils
import appsecrets as appsecrets
import random

art_styles = [
    "Oil painting style with soft brushstrokes and warm tones",
    "Watercolor painting style with flowing colors and organic textures",
    "Pencil sketch style with detailed shading and lifelike textures",
    "Charcoal sketch style with dramatic contrast and gritty textures",
    "Realistic digital painting style with precise detailing and vibrant colors",
    "Acrylic painting style with bold brushstrokes and vivid colors",
    "Photorealistic digital art style with accurate lighting and shading",
    "Pen and ink style with intricate linework and organic textures"
]

openai.api_key = appsecrets.OPEN_AI_API_KEY

def create_story_and_scenes( story, media_url ):
    # save a hard copy of the story
    story_file_path=os.path.join('src', 'outputs', 'story_output.txt')
    utils.save_file(story_file_path, story)
    # turn story into scenes
    file_path = os.path.join("src", "input_prompts", "scenes.txt")
    storyscene = utils.open_file(file_path).replace('<<STORY>>', story)
    gpt_story_scene = gpt3_story_scene(storyscene)
    # Save each scene in its own file
    story_scene_path=os.path.join('src', 'output_story_scenes', 'storyscene.txt')
    utils.save_file(story_scene_path, gpt_story_scene)
    split_story_into_scenes(story_scene_path)

    scenes = [utils.open_file(os.path.join('src', 'output_story_scenes', 'scene1.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene2.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene3.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene4.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene5.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene6.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene7.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene8.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene9.txt')),
              utils.open_file(os.path.join('src', 'output_story_scenes', 'scene10.txt'))]
    
    # Turn our scenes into AI prompts
    count = 0
    mjv4_output_path=os.path.join('src', 'output_story_scenes', "mjv4_output.txt")
    utils.save_file(mjv4_output_path, '')
    selected_style = random.choice(art_styles)
    for scene in scenes:
        count += 1    
        file_path = os.path.join("src", "input_prompts", "mjv4prompts.txt")
        mjv4 = utils.open_file(file_path).replace('<<SCENE>>', scene)
        mjv4_prompt = mjv4.replace('<<PROMPT>>', selected_style)
        desc = gpt3_story_scene(mjv4_prompt)

        # and write to the same file
        current_file = open(mjv4_output_path, 'a')
        if (count == 1):
            current_file.write(desc)
        else:
            current_file.write('\n' + desc)
        current_file.close()

        if count > 10:
            return file_path

def split_story_into_scenes(story_file_path):
  # Open the story file
  folder_path=os.path.join('src', 'output_story_scenes')
  with open(story_file_path, "r", encoding='UTF-8') as story_file:
    # Read the entire file into a single string
    story = story_file.read()
    
  # Split the story into a list of scenes, using the word "Scene" as the delimiter
  scenes = story.split("Scene")

  # Iterate over the list of scenes
  for i, scene in enumerate(scenes):
    # Write each scene to a separate file
    if (i > 0):
        scene_path=os.path.join(folder_path, f'scene{i}.txt')
        with open(scene_path, "w", encoding='UTF-8') as scene_file:
            scene_file.write(scene)
        
def gpt3_story_scene(
    prompt, 
    engine='text-davinci-003', 
    temp=0.7, 
    top_p=1.0, 
    tokens=2000, 
    freq_pen=0.0, 
    pres_pen=0.0, 
    stop=['asdfasdf', 'asdasdf']
):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
                
            text = response['choices'][0]['text'].strip()
            return text
        except Exception as oops:
            print(oops)
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
