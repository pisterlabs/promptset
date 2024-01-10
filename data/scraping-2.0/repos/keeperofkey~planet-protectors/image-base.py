# coding: utf-8
import os
import numpy as np
import pandas as pd
import openai
import requests
from PIL import Image, ImageDraw, ImageFont
import random
import time
import json

#possible intergration into Instagram
#from instabot import Bot
#bot = Bot()
#bot.login(username="", password="")
i=0

openai.api_key = os.getenv("OPENAI_API_KEY")
while i < 25:
    concept = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate a summary of a study about climate change:",
      temperature=0.8,
      max_tokens=32,
      top_p=1,
      frequency_penalty=1,
      presence_penalty=2,
      stop="."
    )
    print(concept.choices[0].text.strip() + "\n")
    
    climate_concept = concept.choices[0].text.strip("\n")
    #this generates the prompt for DALL-E needs context/concept first
    scene = openai.Completion.create(
      model="text-davinci-002",
      prompt="Here is a concept:\n" + climate_concept + "\n" + "Describe an image that conveys this:",
      temperature=1,
      max_tokens=128,
      top_p=1,
      frequency_penalty=1,
      presence_penalty=2,
      echo=True
    )
    print(scene.choices[0].text.strip() + "\n")
    img_desc = scene.choices[0].text.strip("\n")
    #this writes the caption for the image described not sure impact of order here
    caption = openai.Completion.create(
      model="text-davinci-002",
      prompt=img_desc + "Write a funny caption about this:",
      temperature=1,
      max_tokens=32,
      top_p=1,
      frequency_penalty=1,
      presence_penalty=2
    ) 
    print(caption.choices[0].text + "\n")
    #sprinkle of style
    #lighting = ["Golden Hour", "Blue Hour", "Midday", "Overcast"]
    #mood = ["Gothic","Surreal","Cyberpunk","Afrofuturism"]
    #styles = ["Painting","Photograph","Rendering","Experimental","Drawing","Sculpture",]
    #style = random.choice(styles) + ", " + random.choice(lighting) + ", " + random.choice(mood)
    #print(style)
    #prompt = scene.choices[0].text.strip() + "," + style
    #this calls DALLE to gen a single image
    image = openai.Image.create(prompt=img_desc, n=1, size="1024x1024")
    print(image.data[0].url)
    
    #this formats caption could have more cases
    f_caption = caption.choices[0].text.strip("\n")
    if len(f_caption) > 69:
        s = f_caption[:70] + "\n" + f_caption[70:]
        f_caption = s
    elif len(f_caption) > 138:
        sen = f_caption.split(" ")
        s = f_caption[:2]
        f_caption = s
    #this downloads image locally
    f_img = "./images/{}.jpg".format(f_caption)
    img_data = requests.get(image.data[0].url).content
    with open(f_img, 'wb') as handler:
        handler.write(img_data)
    #this reopens image for edits, selects font, draws caption box, writes caption, saves
    img = Image.open(f_img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('SourceCodePro-Black.otf', 24)
    draw.line((0,992) + (1024,992), fill=0, width=64)
    draw.text((0, 960), "{}".format(f_caption), fill=(255, 0, 0), font=font)
    img.save(f_img)
    #bot.upload_photo(f_img, caption=concept.choices[0].text)
    #this creates dataframe by ziping lists together
    #list_of_tuples = list(zip(climate_concepts, image_descriptions))
    #climate_change_images_df = pd.DataFrame(list_of_tuples, columns = ['prompt', 'completion'])
    #climate_change_images_df.head()
    
    #write to json
    obj = {"prompt": climate_concept, "completion": img_desc}
    df_json = json.dumps(obj)
    print(df_json)
    f_data = "./data/data.jsonl"
    with open(f_data, 'a') as handler:
        handler.write(df_json + "\n")
    i += 1
####Export to Excel###
#file_import_location = './data/'
##Create a Pandas Excel writer using XlsxWriter as the engine.
#writer = pd.ExcelWriter(file_import_location + 'climate_change_gpt3_image_data.xlsx', engine='xlsxwriter')
#
## Write each dataframe to a different worksheet.
#climate_change_images_df.to_excel(writer, sheet_name='Image_Data', index=False)
#
#
## Close the Pandas Excel writer and output the Excel file.
#writer.save()
