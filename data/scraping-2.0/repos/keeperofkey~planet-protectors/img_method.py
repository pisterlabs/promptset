# coding: utf-8
import os
import openai
import requests
from PIL import Image, ImageDraw, ImageFont
#export OPENAI_API_KEY=your-api-key-here #put in .bashrc or similar or run before in term
openai.api_key = os.getenv("OPENAI_API_KEY")
#maybe hard code concept
#concept = openai.Completion.create(
#  model="text-davinci-002",
#  prompt="Generate a concept related to climate change:\n",
#  temperature=1,
#  max_tokens=64,
#  top_p=1,
#  frequency_penalty=1,
#  presence_penalty=1
#)
#print(concept.choices[0].text)
concept = "Climate change will result in environmental degradation, increased extreme weather conditions, and decreased resource availiblity. This will cause risks to human and ecosystem health and the viablity of infrastucture"
class img_gen(concept):
    def __init__(self):
        self.self = self
    def dalle_prompt(self):
        self.scene = openai.Completion.create(
          model="text-davinci-002",
          prompt="Describe a funny image representing this:\n" + concept,
          temperature=1,
          max_tokens=128,
          top_p=1,
          frequency_penalty=1,
          presence_penalty=1
        )
        print(self.scene)
        return self.scene.choices[0].text
    def caption(self):
        self.caption = openai.Completion.create(
          model="text-davinci-002",
          prompt="Write a funny caption about this:\n" + dalle_prompt,
          temperature=1,
          max_tokens=64,
          top_p=1,
          frequency_penalty=1,
          presence_penalty=1
        ) 
        return caption.choices[0].text.strip("\n")
    def dalle_img(self):
        self.image = openai.Image.create(prompt=self.scene, n=1, size="1024x896")
        return self.image.data[0].url
    img_data = requests.get(dalle_img(concept)).content
    with open("./images/{}.jpg".format(caption), 'wb') as handler:
        handler.write(img_data)
    img = Image.open("./images/{}.jpg".format(caption))
    img_resize = Image.resize("1024x1024", img)
    draw = ImageDraw.Draw(img_resize)
    font = ImageFont.truetype('SourceCodePro-Black.otf', 24)
    draw.text((0, 1024), "{}".format(caption), fill=(255, 0, 0), font=font)
    img.save("./images/{}.jpg".format(caption))
    
