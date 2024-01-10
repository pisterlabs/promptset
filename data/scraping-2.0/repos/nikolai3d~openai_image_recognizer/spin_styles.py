from openai import OpenAI
import os
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import time

def save_image_from_url(url, save_path):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(save_path)

def spin_styles_sync(i_prompt, i_folder_path):

    # Create a folder if it does not exist
    print (f"Creating folder {i_folder_path}...")

    Path(i_folder_path).mkdir(parents=True, exist_ok=True)
    


    # Create an OpenAI client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # List of styles
    styles = [
        'Abstract Art', 'Abstract Geometry', 'Art Deco', 'Art Nouveau', 'Bauhaus', 
        'Bokeh Art', 'Brutalism in design', 'Byzantine Art', 'Celtic Art', 'Charcoal', 
        'Chinese Brush Painting', 'Chiptune Visuals', 'Concept Art', 'Constructivism', 
        'Cyber Folk', 'Cybernetic Art', 'Cyberpunk', 'Dadaism', 'Data Art', 
        'Digital Collage', 'Digital Cubism', 'Digital Impressionism', 'Digital Painting', 
        'Double Exposure', 'Dreamy Fantasy', 'Dystopian Art', 'Etching', 'Expressionism', 
        'Fauvism', 'Flat Design', 'Fractal Art', 'Futurism', 'Glitch Art', 'Gothic Art', 
        'Gouache', 'Greco-Roman Art', 'Impressionism', 'Ink Wash', 'Isometric Art', 
        'Japanese Ukiyo-e', 'Kinetic Typography', 'Lithography', 'Low Poly', 'Macabre Art', 
        'Magic Realism', 'Minimalism', 'Modernism', 'Monogram', 'Mosaic', 'Neon Graffiti', 
        'Neon Noir', 'Origami', 'Papercut', 'Parallax Art', 'Pastel Drawing', 'Photorealism', 
        'Pixel Art', 'Pointillism', 'Polyart', 'Pop Art', 'Psychedelic Art', 'Rennaissance/Baroque', 
        'Retro Wave', 'Romanticism', 'Sci-Fi Fantasy', 'Scratchboard', 'Steampunk', 'Stippling', 
        'Surrealism', 'Symbolism', "Trompe-l'eil", 'Vaporwave', 'Vector Art', 'Voxel Art', 
        'Watercolor', 'Woodblock Printing', 'Zen Doodle'
    ]


    # Loop through the styles
    rate_limit_delay_sec = 1
    for style in styles:

        # Wait 15 seconds:
        print (f"Waiting for {rate_limit_delay_sec} seconds to avoid rate limit...")

        time.sleep(rate_limit_delay_sec)

        print (f"Generating image in the style of {style}...")

        styled_prompt = i_prompt + ", in the style of " + style

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=styled_prompt,
                size="1024x1024",
                quality="standard",
                n=1)

            image_url = response.data[0].url

            # Download the image
            destination_path = f"{i_folder_path}/{style}.png"
            print (f"Downloading and saving image...")

            save_image_from_url(image_url, destination_path)
            print (f"Image {destination_path} saved.")
            
        except Exception as e:
            print (f"Error generating image in style of {style}: {e}")

        

# spin_styles_sync("Multiple Kittens of various breeds, playing in and around a Christmas tree", "./christmas_kittens_hd")
# spin_styles_sync("Idyllic Lake in the mountains, with a small village on its bank, during winter", "./village_lake_hd")
# spin_styles_sync("Santa Claus driving a sleigh, reindeer pulling it, taking off from his base at the North Pole", "./santa_hd")
# spin_styles_sync("Multiple gifts under a christmas tree, with two grey tabby cats sleeping next to them", "/home/nikolai3d/Dropbox/AdobeFirefly/Style Spin/cat_gifts_hd")
# spin_styles_sync("Three young pretty witches in a winter forest celebrating Christmas", "/home/nikolai3d/Dropbox/AdobeFirefly/Style Spin/witches_hd")
# spin_styles_sync("Human developer and robot developer, pair-programming at home at Christmas Eve", "/home/nikolai3d/Dropbox/AdobeFirefly/Style Spin/pair_programming_hd")
# spin_styles_sync("A bustling Santa's Workshop scene, massive in scale, filled with festive cheer and holiday spirit. The workshop is adorned with colorful Christmas decorations, twinkling lights, and a large, ornate Christmas tree. Santa Claus, an elderly Caucasian man with a jovial expression and a traditional red and white suit, oversees the activities. Around him, a diverse group of elves, wearing green and red outfits, are energetically crafting toys and gifts. The elves vary in gender and descent. The workshop is a whirlwind of activity, with elves painting toys, wrapping gifts, and checking lists, all set against a backdrop of snowy windows and a cozy fireplace.", "/home/nikolai3d/Dropbox/AdobeFirefly/Style Spin/santa_workshop_hd")
            
spin_styles_sync("Christmas", "/home/nikolai3d/Dropbox/AdobeFirefly/Style Spin/christmas_sd")