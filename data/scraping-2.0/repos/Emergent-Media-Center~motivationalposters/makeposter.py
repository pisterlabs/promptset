import json
import requests
import io
import base64
import openai
import argparse
import textwrap
from PIL import Image, PngImagePlugin, ImageFont, ImageDraw

sd_url = "http://127.0.0.1:7860"

argParser = argparse.ArgumentParser()
argParser.add_argument('-p', '--prompt', help = 'the main word for the poster', default = 'Prompt')
argParser.add_argument('-d', '--demotivate', help = 'makes a demotivational instead of a motivational', action = 'store_true')

args = argParser.parse_args()


demotivate = args.demotivate
keyword = args.prompt

apifile = open('apikey.txt')
openai.api_key = apifile.read()




def generateimage(apitype,payload):
    response = requests.post(url=f'{sd_url}/sdapi/v1/' + apitype, json=payload)
    r = response.json()
    if 'images' in r:
        return r['images'][0]
    else:
        print('ERROR!')
        print(r)

def saveb64(imgb64, filename=None):
    image = Image.open(io.BytesIO(base64.b64decode(imgb64.split(",",1)[0])))
    if filename != None:
        image.save(filename)
    return image
    

def loadb64(filename):
    img = Image.open(filename)
    imgfile = io.BytesIO()
    img.save(imgfile, format='PNG')
    imgb64 = base64.b64encode(imgfile.getvalue()).decode("utf-8") 
    #imgb64 = imgb64[2:-1]
    return imgb64
    
def chatgpt(m):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = m
    )
    return response.choices[0].message.content

def motivationalprompt():
    sysmessage = ("Your job is to generate motivational phrases. "
    "A user will enter a short word or phrase, and you must start your phrase with it, along with a colon (:). " 
    "After the phrase, put a $ and enter a description of a photograph that might fit your phrase if it were printed on a poster.")
    return chatgpt([
        {"role": "system", "content":    sysmessage},
        {"role": "user", "content":      "solution"},
        {"role": "assistant", "content": "Solution: To every problem there is a solution, whether you know it or not.$Two hands trying to fit puzzle pieces together"},
        {"role": "user", "content":      "leaders"},
        {"role": "assistant", "content": "Leaders: One without any followers is just someone taking a walk.$Penguins in the arctic all marching in a line"},
        {"role": "user", "content":      keyword}
    ])
    
def demotivationalprompt():
    sysmessage = ("Your job is to generate mean-spirited snarky demotivational phrases. "
    "A user will enter a short word or phrase, and you must start your phrase with it, along with a colon (:). " 
    "After the phrase, put a $ and enter a description of a photograph that might fit your phrase if it were printed on a poster.")
    return chatgpt([
        {"role": "system", "content":    sysmessage},
        {"role": "user", "content":      "paranoia"},
        {"role": "assistant", "content": "Paranoia: If it feels like everyone is out to get you, they probably are.$A woman in a trench coat with spy goggles"},
        {"role": "user", "content":      "practice"},
        {"role": "assistant", "content": "Practice: No matter how much you do it, you'll probably never be that good.$A football player trying to catch a ball as it flies through the air"},
        {"role": "user", "content":      keyword}
    ])
    
sdprompt = "Large red text reading ERROR"
bottomtext = "We all make mistakes!"

result = "placeholder: holds your place.$a placeholder"
if not demotivate:
    result = motivationalprompt()
else:
    result = demotivationalprompt()

result_split = result.split("$",1)
sdprompt = result_split[1].strip()
secondsplit = result_split[0].split(":",1)
bottomtext = secondsplit[1].strip()


keyword = keyword.upper()
bottomtext = bottomtext.upper()
if bottomtext[-1] == ".":
    bottomtext = bottomtext[:-1]
sdprompt = sdprompt.lower()
sdprompt = sdprompt.replace(".","")
sdprompt = sdprompt + " photograph realistic"

print(keyword)
print(bottomtext)
print(sdprompt)

imageb64 = generateimage('txt2img', {
    "prompt": sdprompt,
    "steps": 20,
    "width": 512,
    "height": 320,
    "override_settings": {
        "sd_model_checkpoint": "v1-5-pruned-emaonly.safetensors [6ce0161689]"
    }
})
genimage = saveb64(imageb64, 'genimage.png')

poster = Image.open('template.png')
poster.paste(genimage,(69,30))


font_large = ImageFont.truetype(font='NEWBASKE.ttf', size=64)
font_small = ImageFont.truetype(font='NEWBASKE.ttf', size=22)
posterdraw = ImageDraw.Draw(poster)

posterdraw.text((325,410),keyword,anchor='ms',font=font_large)

wrappedtext = textwrap.wrap(bottomtext,width=46)
y = 435
for i in wrappedtext:
    posterdraw.text((325,y),i,anchor='ms',font=font_small)
    y = y + 20

poster_filename = keyword.lower() 
poster_filename = poster_filename + ".png"
poster.save(poster_filename)

