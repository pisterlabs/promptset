
import openai
import os
from PIL import ImageDraw, ImageFont, Image
from pathlib import Path
from base64 import b64decode
from typing import Final
from datetime import datetime
from dotenv import load_dotenv
from functions import dataHandler

SECRETS_NAME = 'secrets.env'
DATA_NAME = 'data.json'

# Get path to .env file
DIRPATH = os.path.dirname(__file__)
dotenv_path = os.path.join((os.path.abspath(os.path.join(DIRPATH, ".."))), SECRETS_NAME) 
load_dotenv(dotenv_path)

# Get path to data.json file
data_path = os.path.join((os.path.abspath(os.path.join(DIRPATH, ".."))), DATA_NAME) 

# Load data
data = dataHandler.load_data(data_path)

# Openai API key
openai.api_key = os.getenv("OPENAI_API_KEY")



def save_image(response, file_path):
    now = datetime.now()
    date = now.strftime("%Y%m%d")

    
    for index, image_dict in enumerate(response["data"]):
        image_data = b64decode(image_dict["b64_json"])
        image_path = file_path + "/" + f"{date}.jpeg"
        with open(image_path, mode="wb") as jpeg:
            jpeg.write(image_data)
            
    image = Image.open(image_path)
            
    
    return image, image_path


def generateImage(IMAGES_FOLDER_PATH):
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    image_path = IMAGES_FOLDER_PATH + "/" + f"{date}.jpeg"
    
    if Path(image_path).is_file():
         # Caption creation
        caption_prompt = data["caption_prompt"]
        
        response = openai.Completion.create(
            engine = "text-davinci-002",
            prompt = caption_prompt,
            max_tokens = 50
        )

        caption = response.choices[0].text
        
        if caption[0] == '\n':
            caption = caption[1:]

        
    else:
        
         # Create with gpt the caption
        caption_prompt = data["text_prompt"]
        
        response = openai.Completion.create( #gpt generating
            engine = "text-davinci-002",
            prompt = caption_prompt,
            max_tokens = 50
        )

        text = response.choices[0].text

        split = text.split() #removing too long phrase
        i = 0
        max_word_per_phrase = 25
        text = ""
        for word in split:
            i += len(word)
            if i >= max_word_per_phrase:
                i = 0
                text = text+'\n'+word
            else:
                text = text+' '+word
        text = text[1:]

        # Prompt generating
        image_prompt = data["image_prompt"]

        # Image generating
        response = openai.Image.create(
            prompt=image_prompt,
            model="image-alpha-001",
            size="1024x1024",
            response_format="b64_json"
        )
        
        # Convents and saves image
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        
        img, image_path = save_image(response, IMAGES_FOLDER_PATH)
            

        # Combine text and image
        max_font_size = 50
        font = ImageFont.truetype('Optima.ttc', max_font_size)
        draw = ImageDraw.Draw(img)
        textwidth, textheight = draw.textsize(text, font)
        width, height = img.size
        x=width/2-textwidth/2
        y=height/2-textheight/2

        while (x < 5) :
            max_font_size -= 1
            font = ImageFont.truetype('Optima.ttc', max_font_size)
            draw = ImageDraw.Draw(img)
            textwidth, textheight = draw.textsize(text, font)
            width, height = img.size
            x=width/2-textwidth/2
            y=height/2-textheight/2
                
        draw.text((x, y), text, font=font, fill='white')
        
        
        img.save(image_path)
        

        # Caption creation
        caption_prompt = data["caption_prompt"]
        
        response = openai.Completion.create(
            engine = "text-davinci-002",
            prompt = caption_prompt,
            max_tokens = 50
        )

        caption = response.choices[0].text
        
        if caption[0] == '\n':
            caption = caption[1:]
        
    print(image_path, caption)
        
        
    return image_path, caption

async def post_image(bot, image_path, caption):
    # Publication post
    bot.upload_photo(image_path, caption=caption)
    
'''
# For testing
if __name__ == '__main__':
    image_path, caption = generateImage(os.path.join((os.path.abspath(os.path.join(DIRPATH, ".."))), "image") )
'''