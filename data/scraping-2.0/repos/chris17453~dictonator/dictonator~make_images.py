import re
import os
import openai
import sqlite3
import requests

# Retrieve the API Key from environment variables
api_key = os.getenv('dict_openai_api_secret')

if api_key is None:
    raise ValueError("API_KEY is not set in the environment variables")

# Set up OpenAI with the API Key
openai.api_key = api_key

def image(word,prompt_text):
    response = openai.Image.create(
    prompt=prompt_text,
    n=1,
    size="512x512"
    )
    image_url = response['data'][0]['url']

    # Download and save the image locally
    response = requests.get(image_url)

    file=os.path.join('images',word+".png")
    with open(file, 'wb') as f:
        f.write(response.content)
    


# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('dictionary_entries.db')
# Create a new SQLite cursor
cur = conn.cursor()

# Retrieve data from the database
cur.execute('SELECT definition,word,synonyms,base_word,image_scene_prompt FROM dictionary_entries ORDER BY word ASC')
rows = cur.fetchall()


def strip_unsafe_characters(word):
    """
    Remove all characters that are not alphanumeric, hyphens, or underscores from a string.

    Parameters:
    - word (str): The input string.

    Returns:
    - str: The sanitized string.
    """
    # Regular expression pattern to match any character that is NOT alphanumeric, hyphen or underscore
    pattern = re.compile(r'[^\w-]')
    # Replace all matches of the pattern with an empty string
    return pattern.sub('', word)



def gpu2(word,prompt):
    import json
    import requests
    import io
    import base64
    from PIL import Image

    url = "http://gpu2.watkinslabs.com"

    payload = {
        "prompt": prompt,
        "steps": 25,
        "width":1024,
        "height":256,
        "negative_prompt":"((cropped image, out of shot, cropped)),worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    safe_word=strip_unsafe_characters(word)
    file=os.path.join('images',safe_word+".jpg")
    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save(file)    

print("Looping")
prompts=[
    #"{prompt}, wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols"
    "a colorfull bubbly tech panel, {prompt} as inkpunk",
    "{prompt}, anthro, very cute kid's film character, disney pixar zootopia character concept artwork, 3d concept, detailed fur, high detail iconic character for upcoming film, trending on artstation, character design, 3d artistic render, highly detailed, octane, blender, cartoon, shadows, lighting",
    "cinematic photo portrait of {prompt}, cyberpunk, in heavy raining futuristic tokyo, rooftop, cyberpunk night, sci-fi, fantasy, intricate, extremely beautiful, elegant, neon light, highly detailed, digital painting, artstation, concept art, soft light, hdri, smooth, sharp focus, illustration, art by tian zi and craig mullins and wlop and alphonse mucha . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    #"Digital Art {prompt}, ultra realistic, concept art, intricate details, highly detailed, photorealistic, octane render, 8k, unreal engine, sharp focus, volumetric lighting unreal engine, art by artgerm and alphonse mucha",
    #"cinematic photo {prompt} drawing on canvas, ((drawn by a kid)), child's drawing, ((unfinished drawing)), canvas, crayons laying on the desk, 35mm photograph, film, bokeh, professional, 4k, highly detailed",
#    "portrait photo of {prompt}, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "23rd century scientific schematics for {prompt}, blueprint, hyperdetailed vector technical documents, callouts, legend, patent registry"
]
index=0
for row in rows:
    pre_prompt=row[4]
    prompt=prompts[index].format(prompt=pre_prompt)
    index=index+1
    index=index%len(prompts)
    print(prompt)
    word=row[1]
    gpu2(word,prompt)
