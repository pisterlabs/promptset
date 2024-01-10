import asyncio
import base64
import hashlib
import html
import io
import json
import os
import re
import signal
import time
import traceback
import urllib
import urllib.request
import warnings
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from random import randrange
from threading import Semaphore, Thread

import discord
import easyocr
import nest_asyncio
import numpy as np
import requests
import torch
import urllib3
from duckduckgo_search import ddg
from mathjson_solver import create_solver
from openai import OpenAI
from PIL import Image
from transformers import (AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer,
                          Blip2ForConditionalGeneration,
                          MusicgenForConditionalGeneration)

from comfyui import generate
from inference import infer
from prompts import getsdprompts

script_dir = Path(os.path.abspath(__file__)).parent

class Config:
    def __init__(self):
        with open(os.path.join(script_dir, 'config.json')) as config_file:
            config = json.load(config_file)

        self.backend = config['Backend']
        self.host = config['HOST']
        self.port = config['PORT']
        self.whitelisted_servers = config['Whitelisted_servers']
        self.base_context = config['Base_Context']
        self.model_name = config['model_Name']
        self.enabled_features = config['Enabled_features']
        self.llm_parameters = config['LLM_parameters']
        self.adminid = config['admin_id']
        self.token = config['token']
        self.api_key = config['api_key']
        self.commands = config['commands']

    def check_command(self, message, id):
        for command in self.commands:
            if command['name'] in message and id == self.adminid:
                return command
        return None

        print("Loaded config")

config = Config()

openaiclient = OpenAI(base_url = f"http://{config.host}:{config.port}/v1", api_key=config.api_key,) 

#os.environ["NVIDIA_VISIBLE_DEVICES"] = "2" #Change/uncomment as needed. Mostly here for my use as i find it more convenient.
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if config.enabled_features['image_input']['enable']:
    yolo = torch.hub.load("ultralytics/yolov5", "yolov5x")  # or yolov5m
    reader = easyocr.Reader(['en'])
    if config.enabled_features['image_input']['mode'] == 'blip':
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float32)
        device = "cpu"
        model.to(device)
if config.enabled_features['music_generation']:
    musicprocessor = AutoProcessor.from_pretrained("facebook/musicgen-small")

    musicmodel = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
serverwhitelist = []
today = date.today()

seed = -1
obj = Semaphore()
vcsemp = Semaphore()
imgsem = Semaphore()


basecontexx = config.base_context
contexx = basecontexx

def remove_unicode(string):
    nm = ''.join(filter(lambda c: ord(c) < 128, string))
    if nm == '':
        nm = f'unicodename{randrange(1, 4000)}'
    return nm


history = []
try:
    with open(os.path.join(script_dir, 'memory.json'), 'r') as f:
        history = json.load(f)
except:
    with open(os.path.join(script_dir, 'memory.json'), 'w') as f:
        json.dump(history, f)

usrlist = ['\n###', '<\s>', '<s>', '[INST]', '[/INST]', '[INST', 'User:', '</s>', '<|im_end|>']
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
lasttime = round(time.time())
knownusrs = []

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

pos = 0
flanmodel = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
flantokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model_name = config.model_name



# Globals
nest_asyncio.apply()
client = discord.Client(enable_debug_events=True)
prev = ''
globtext = ''
queue = 0
bloommem = []
imagemaking = False
blipcache = {}
imagemem = {}
blacklist = []
dmlist = {}
lastans = ""
raw = False
Debugimg = False
enableimg = config.enabled_features['image_generation']['enable']
sdmem = {}
temp = 0.7
msgqueue = [False for i in range(999)]
ignore = False


procs = False


def handler(signum, frame):
    global temp
    res = input("Ctrl-c was pressed. Type exit to quit. ")
    match res.lower().split(" ")[0]:
        case "exit":
            exit(1)
        case _:
            print("continuing.")
            return

def aspect2res(inp): #GETS TRIGGERED BY "TUXEDO" MUST FIX <- Been here for weeks. Doesnt really affect anything so :shrug: Also comfyui appears to not work with it for whatever reason.
    aspect = ""
    for x in inp.split(","):
        if "x" in x or ":" in x:
            for p in x.split(" "):
                if "x" in p or ":" in p:
                    aspect = p
                    print(f"Gotten aspect: {p}" )
                    break
    aspect = aspect.replace(":", "x").replace("1920x1080", "16x9")
    aspects = {}
    aspects['16x9'] = ["1365","768"]
    aspects['9x16'] = ["768","1344"]
    aspects['4x3'] = ["1182","886"]
    if not aspect == "":
        try:
            return aspects[aspect]
        except KeyError:
            return ["1024", "1024"]
    else:
        return ["1024", "1024"]
signal.signal(signal.SIGINT, handler)

from collections import Counter


def remove_duplicates(data): #ft. airoboros-l2-70b
  """
  This function removes duplicates from a list and returns a new list with the count of each element.

  Args:
    data: A list of strings.

  Returns:
    A list of strings with the count of each element.
  """
  # Create a dictionary to store the count of each element.
  counts = Counter(data)

  # Create a new list to store the results.
  results = []

  # Loop through each element in the dictionary.
  for key, value in counts.items():
    # If the count is greater than 1, add the element to the results list with the count.
    if value > 1:
      results.append(f"{value}x {key}")
    # If the count is 1, add the element to the results list without the count.
    else:
      results.append(key)

  return results

def GetYoutubeTitle(VideoID):
    title = ''
    params = {"format": "json",
              "url": "https://www.youtube.com/watch?v=%s" % VideoID}
    url = "https://www.youtube.com/oembed"
    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string
    if url:
        print(f'Getting Youtube title from {url}')
    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        title = data['title']
    return title

def get_position(bbox, width, height): #Thanks mixtral, works pretty good
    # Unpack bounding box coordinates
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
    def percentage(percent, whole):
      return (percent * whole) / 100.0

    # Calculate bounding box center
    bbox_center_x = (x1 + x2 + x3 + x4) / 4
    bbox_center_y = (y1 + y2 + y3 + y4) / 4

    # Calculate center of the provided width and height
    center_x = width / 2
    center_y = height / 2

    # Determine the relative position
    if bbox_center_x < center_x and bbox_center_y < center_y:
        return "top-left corner"
    elif abs(bbox_center_x - center_x) < percentage(10, width):
        return "center vertically"
    elif abs(bbox_center_y - center_y) < percentage(10, height):
        return "center horizontally"
    elif bbox_center_x > center_x and bbox_center_y < center_y:
        return "top-right corner"
    elif bbox_center_x < center_x and bbox_center_y > center_y:
        return "bottom-left corner"
    elif bbox_center_x > center_x and bbox_center_y > center_y:
        return "bottom-right corner"
    elif bbox_center_x == center_x and bbox_center_y == center_y:
        return "center"
    else:
        return "unknown position"

def TTS(input_text, output_file, model=config.enabled_features['TTS']['model'], voice=config.enabled_features['TTS']['voice'], response_format="mp3", speed=config.enabled_features['TTS']['speed']):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "input": input_text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed
    }

    response = requests.post(config.enabled_features['TTS']['URI'], headers=headers, json=data)

    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
            print(f"File {output_file} has been created.")
    else:
        print(f"Failed to create file. Status code: {response.status_code}")


async def replier(message, text):
    global lastans
    global globtext
    global Debugimg
    global lastsent
    global imagemem
    global voiceclient
    regex = r";j6487\S*;j6487"
    subst = ""
    if globtext == '':
        print("Not sending.")
        return
    if ';j6487' in text:

        imgdata = text.split(';j6487')[1]
        globtext = ''

        text = re.sub(regex, subst, text, 0, re.MULTILINE)
        fele = discord.File(io.BytesIO(
            base64.b64decode(imgdata)), filename='img.png')
        text = re.sub(r'\\_', '_', text)

        lastans = text
        text = re.sub(r"https?://[\S]+.\w{2,}", '', text)
        text = text.replace("=_", '')
        lastsent = await message.reply(text, file=fele)
        if Debugimg:
            with open('imagedebug.txt', 'w') as f:
                f.write(imgdata)
        print(f'Sending: {text} With a file')
        try:
            imagemem[lastsent.id] = sdmem[lastsent.guild.id]
        except Exception:
            imagemem[lastsent.id] = sdmem[lastsent.guild.id]
            print(traceback.format_exc())
    elif ';jMUSIC62515' in text:
        globtext = ''

        text = re.sub(regex, subst, text, 0, re.MULTILINE)
        with open('musicgen_out.wav', 'rb') as f:
            fele = discord.File(f, filename='audio.wav')
        text = re.sub(r'\\_', '_', text)

        lastans = text
        text = re.sub(r"https?://[\S]+.\w{2,}", '', text)
        text = text.replace("=_", '')
        text = text.replace(";jMUSIC62515",'')
        lastsent = await message.reply(text, file=fele)

        print(f'Sending: {text} With a file')
  
    else:
        try:
            if voiceclient and config.enabled_features['TTS']['enable']: #ancient feature. Hasnt been touched in months.
                vcsemp.acquire()
                TTS(text,'voice.wav')
                while voiceclient.is_playing():
                    pass
                voiceclient.play(discord.FFmpegPCMAudio('./voice.wav'))
                vcsemp.release()
        except NameError:
            pass
        regexcln = r"<\S*>h"
        endcln = r"</.*>"
        text = re.sub(regexcln, 'h', text, 0, re.MULTILINE)
        text = re.sub(endcln, '', text, 0, re.MULTILINE)
        print(f'Sending: {text}')
        globtext = ''
        text = re.sub(r'\\_', '_', text)
        lastans = text
        lastsent = await message.reply(text)


def imagegen(msg, guild, replyid, imgtoimg):
    global sdmem
    global raw
    global imagemaking
    global imagemem
    sample = "1. standing white Persian cat, filmed with a Canon EOS R6, 70-200mm lens, high quality\n2. standing white Persian cat, photo, filmed with a Canon EOS R6, 70-200mm lens, high quality"
   
    payload = getsdprompts(replyid, imagemem, msg, guild, sdmem, imgtoimg)
    chat_completion = openaiclient.chat.completions.create(model="gpt-4", messages=payload, temperature=0.1, max_tokens=150, stop=usrlist)
    
    imagemaking = True
    if raw:
        tosend = msg
        print(f'Prompt: {tosend}')
    else:

        rfn = chat_completion.choices[0].message.content
        obj.release()
        output = re.split(r"\d\.", rfn)
        print(output)
        tosend = ''
        try:

            tosend = list(output)[1].replace('\n', '')
            if len(output) > 2:
                similarity = similar(sample, output[1])
                print(f"Similarity to example: {similarity}")
                if similarity > 0.7:
                    print("Too similar to example.")
                    tosend = output[2].replace('\n','')
            print(f'Prompt: {tosend}')
        except Exception:
            print(f"Error: {traceback.format_exc()}")

            tosend = ''.join(output)
            print(f'Prompt: {tosend}')
            tosend = f'{tosend}'
    if not imgtoimg:

        res = aspect2res(tosend)
        x = generate(tosend,config.enabled_features['image_generation']["server_address"] , width=res[0], height=res[1])[0]
        sdmem[guild] = [tosend, re.sub(r'\\', '', msg)]
    else: #Not currently functional due to ... comfyui reasons
        pass
        '''print("Performing img2img")
        prebuf = io.BytesIO()
        imgtoimg.save(prebuf, format="PNG")
        imgtoimg = base64.b64encode(prebuf.getvalue())
        imgtoimg = imgtoimg.decode('utf-8')
        myobj = {'prompt': tosend , 'cfg_scale': '8',  "init_images": [imgtoimg],"denoising_strength": '0.63',"width": '1024',"height": '1024', 'steps': '50', 'sampler_name':'LMS'}
        x = requests.post(f'http://{HOST}:7860/sdapi/v1/img2img', json=myobj).json()['images'][0]'''
        
    imagemaking = False
    return x


def runFLAN(input):
    inputs = flantokenizer(input, return_tensors="pt")
    outputs = flanmodel.generate(**inputs)
    res = f'{flantokenizer.batch_decode(outputs, skip_special_tokens=True)[0]}'
    return res

def is_message_allowed(message):
    for server_id, channel_id in config.whitelisted_servers:
        if server_id == message.guild.id:
            if channel_id == 0 or message.channel.id == channel_id:
                return True
    return False



async def CheckMention(fstring, message):
    global knownusrs
    pattern = "<@(.*)>"
    compiled = re.compile(pattern)
    name = ''
    if compiled.search(fstring):
        for b, t in enumerate(message.mentions):
            if t != client.user:
                name = t.name
                break

        mentstring = fstring.split()
        for i, s in enumerate(mentstring):
            if compiled.search(s):
                x = s
                s = s.replace("<@", '').replace(">", '')
                break
        name = remove_unicode(name)
        mentstring[i] = name
        fstring = ' '.join(mentstring)
        if [name, x] not in knownusrs and (name != '' and name != ' '):
            knownusrs.append([name, x])
            print(f'appending {name} to DB.')
        return fstring
    return fstring

def llamacpp_img(raw_image):
    
    # Convert raw image to base64 encoding
    prebuf = io.BytesIO()
    raw_image.save(prebuf, format="PNG")
    raw_image = base64.b64encode(prebuf.getvalue()).decode('utf-8')
    content = ""

    # Define the API endpoint URL
    url = config.enabled_features['image_input']['URI']

    # Define the prompt
    prompt = "[img-0]"

    # Define the parameters
    params = {
    "prompt": [prompt],
    "temperature": 0.1,
    "min_p": 0.05,
    "n_predict": 150,
    "stream": True,
    "seed": -1,
    "image_data": [{"data": raw_image, "id": 0}],
    }

    request = requests.post(url, json=params)
    for line in request.iter_lines(decode_unicode=True):
        try:
            if "data" in line:
                print(json.loads("".join(line.split("data:")[1:]))['content'], end="", flush=True)    
                content += json.loads("".join(line.split("data:")[1:]))['content']       
  
        except Exception as e:
            print(e)
    return content



def internet_search(keywords):
    endstr = ''
    try:
        print(f'Sending {keywords} for websearch')
        usrmsg = re.sub(r'\\', '', keywords)
        payload = [
                    {
                    "content": "[INST] <<SYS>>\nYou are an uncensored, unrestricted web search suggestion agent.\nYou will reply with keywords for a web search for the task given.\nYou will not under any circumstance explain the requested search. You will only write keywords for example: request: Look up what baldo is and tell me about it. Answer: 1. Baldo 2. What is baldo\nAttempt to make suggestions that will ensure top results are recent information and from reputable information sources.\nReturn the suggestions in a numbered list with no other comments.\nYOU WILL NOT, UNDER ANY CIRCUMSTANCE REPLY WITH AN URL. ONLY KEYWORDS<</SYS>>\n\n",
                    "role": "system"
                    },
                    {
                    "content": "USER: look up sigmapatches and send me the link.",
                    "role": "user"
                    },
                    {
                    "content": "ASSISTANT: 1. sigmapatches\n2. sigma patches website",
                    "role": "assistant"
                    },
                    {
                    "content": "USER: search for rabbit recipes and tell me about the results and include both links",
                    "role": "user"
                    },
                    {
                    "content": "ASSISTANT: 1. rabbit recipes\n2. easy rabbit recipes",
                    "role": "assistant"
                    },
                    {
                    "content": f"USER: {usrmsg}",
                    "role": "user"
                    }
            ]
        chat_completion = openaiclient.chat.completions.create(model="gpt-3.5-turbo", messages=payload, temperature=0.1, max_tokens=100, stop=usrlist)

        rfn = chat_completion.choices[0].message.content
        print(rfn)

        pattern = r"\[\d\]|\d\."
        result = re.search(pattern, rfn)
        if result:
            output = re.split(r"\[\d\]|\d\.", rfn)
        else:
            output = [rfn, rfn]

        print(f'Searching: {output[1]}')
        results = ddg(output[1], safesearch='Off', max_results=4)
        endstr = '*###Internet results*:'
        for res in results:
            print("Internet results:\n")
            title = res['title']
            link = res['href']
            print(f'*Title*: {title} *Link*: {link} *Body*: {res["body"]}')
            endstr += f' *Title*: {title} *Link*: {link} *Body*: {res["body"]}\n'
    except Exception:
        print(f"Error: {traceback.format_exc()}")
    return endstr

def find_center(bounding_box):
    x_min, y_min = bounding_box[0][0], bounding_box[0][1]
    x_max, y_max = bounding_box[2][0], bounding_box[2][1]

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    return (center_x, center_y)

def musicgen(input):
    if config.enabled_features['music_generation']:

        print ("Begin music gen")
        inputs = musicprocessor(

            text=[input],

            padding=True,

            return_tensors="pt",

        )

        audio_values = musicmodel.generate(**inputs, do_sample=True, guidance_scale=4, max_new_tokens=350)
        import scipy

        sampling_rate = musicmodel.config.audio_encoder.sampling_rate

        scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
        print("Music written to file")
        return ";jMUSIC62515"
    return ""


def youtubechk(fstring):
    if re.search(r'\byoutube.com/*\b', fstring) or re.search(r'\byoutu.be/*\b', fstring):
        youtubestr = fstring.split()
        title = ''
        for i, s in enumerate(youtubestr):
            if re.search(r'\byoutube.com/*\b', s) or re.search(r'\byoutu.be/*\b', s):
                s = s.replace("youtube.com/watch?v=", '').replace("youtu.be/", '').replace(
                    "<", '').replace(">", '').replace("www.", '').replace("https://", '')
                title = GetYoutubeTitle(s)
                print(title)

        youtubestr[i] = f'*video titled: "{title}"*'
        fstring = ' '.join(youtubestr)
    return fstring


def getllamadesc (dats, messagef, mathjson=False):
    llamadesc = ""
    if mathjson:
        loc = 0
        for i, x in enumerate(re.split(r'\s', dats)):
            if loc == 0:
                if "<mathjson>" in x:
                    loc = i
                    print(loc)
        llamadesc = ' '.join(re.split(r'\s', dats)[loc:])

        return [llamadesc.replace("<mathjson>",'').replace("</mathjson>",''), llamadesc]
    else:
        pattern = r"<(\w+)>(.*?)</\1>"

        matches = re.findall(pattern, dats)
        contents = [match[1] for match in matches]
        print(contents)
        if len(contents) > 0:
            print(f"Got match {contents[0]}")
            llamadesc = contents[0]
        if re.search(r"https?://[\S]+.\w{2,}", dats) or llamadesc =="":
            return messagef
        else:
            return messagef + " *Request summary: " + llamadesc + "*"
def send_message(message):
    response = requests.post(f"http://{config.host}:{config.port}", data={"input": message})
    return json.loads(response.text)["output"]
async def run(message, checker, pos):
    start_time = 0
    start_time = time.perf_counter()
    global dmlist
    global imagemaking
    try:
        guild = message.guild.name
        guildid = message.guild.id
    except AttributeError:
        guildid = 0
        guild = "PrivateMessage"
        if imagemaking:
            waiter = 45
        else:
            waiter = 5
        if message.author.id not in dmlist:  #Anti DM Spam
            print(f"Adding {message.author.id} to dm spam protection.")
            dmlist[message.author.id] = [999,0,False]
        if abs(round(time.time()) - dmlist[message.author.id][0]) < waiter:
            dmlist[message.author.id][1] += 1
            print(f"Striked: {message.author.name}, Strikes at: {dmlist[message.author.id][1]}, Delay: {abs(round(time.time()) - dmlist[message.author.id][0])}")

        dmlist[message.author.id][0] = round(time.time())

    global lasttime
    global model_name #Really need to remove all these globals..
    global knownusrs
    global lastmsg
    global truncation_length
    global globtext
    global queue
    global contexx
    global blacklist
    global enableimg
    global temp
    global msgqueue
    global history
    global messag
    global usrlist
    global procs
    global device
    global lastmsgd
    global lastsent
    global seed
    if guild == "PrivateMessage":
        if dmlist[message.author.id][1] > 4 and dmlist[message.author.id][2] == False:

            globtext = 'Blocked for spam.'
            print(f"Blocked {message.author.name} for DM Spam.")
            messag = message
            dmlist[message.author.id][2] = True
            return
        elif dmlist[message.author.id][2] == True:
            return

    lasttime = round(time.time())
    imageoutput = ""
    ocrTranscription = ""
    ocrTranscriptionT = ""
    foundobjt = ''
    foundobj = ''
    msgqueue[pos + 1] = True
    fstring = message.content
    fstring = fstring.replace(f'<@{client.user.id}>', '')
    raw_image = ''
    try:
        fstring = youtubechk(fstring)
        try:
            image = message.attachments[0].url
            if (".png?" in image or ".jpg?" in image or ".webp?" in image) and config.enabled_features['image_input']['enable']: # Just to be safe.
                raw_image =  Image.open(requests.get(image, stream=True).raw)
                width = raw_image.width 
                height = raw_image.height 
  
                out = ''
                avgimg = raw_image.resize((10, 10), Image.LANCZOS).convert("L")
                pixel_data = list(avgimg.getdata())
                avg_pixel = sum(pixel_data)/len(pixel_data)
                raw_image = raw_image.convert('RGB')
                ocrresult = reader.readtext(np.array(raw_image), paragraph=True)
                ocrTranscription = '"'
                yoloresults = yolo(np.array(raw_image))
                tempres = []
                for x in json.loads(yoloresults.pandas().xyxy[0].to_json(orient="records")):
                    if x['confidence'] > 0.4:
                        print (f"Confidence: {x['confidence']}, {x['name']}")
                        tempres.append(x['name'])
                tempres = remove_duplicates(tempres)
                foundobjt = ",".join(tempres)
                if not foundobjt == "":
                    foundobj = "Object recognition: " + foundobjt
                for x in ocrresult:
                    position = find_center(x[0])
                    ocrTranscriptionT +=  f"'{x[1]}' Position: {position}." + '\n'
                if not ocrTranscriptionT == "":
                    ocrTranscription = "OCR Output: " + ocrTranscriptionT 
                ocrTranscription = ocrTranscription.strip() 
                ocrTranscription += '"'
                print(ocrTranscription)
                bits = "".join(['1' if (px >= avg_pixel) else '0' for px in pixel_data])
                hex_representation = str(hex(int(bits, 2)))[2:][::-1].upper()
                sha = hashlib.sha1(hex_representation.encode()).hexdigest()
                if sha in blipcache:
                    out = blipcache[sha]
                else:
                    if config.enabled_features['image_input']['mode'] == 'blip':
                        inputs = processor(raw_image, return_tensors="pt").to(
                            torch.device(device), torch.float32)
                        generated_ids = model.generate(**inputs, max_new_tokens=100)
                        out = processor.batch_decode(
                            generated_ids, skip_special_tokens=True)[0].strip()
                    else:
                        imgsem.acquire()
                        out = llamacpp_img(raw_image)
                        imgsem.release()
                    blipcache[sha] = out
                imageoutput = out
            elif image.endswith(".pdf"):
                #From when i thought i would make it be able to work with pdfs
                pass
        except Exception:
            print(traceback.format_exc())
            imageoutput = ""
        fstring = await CheckMention(fstring, message)
        print(
            f'Message from {message.author}: {fstring} with {message.attachments} in {guild}')
        print(imageoutput)
        tempmessagef = ''
        if config.enabled_features['internet_search'] and ('search' in fstring.lower() or 'internet' in fstring.lower() or 'lookup' in fstring.replace(" ", '').lower() or 'look up' in fstring.lower() or 'look it up' in fstring.lower()) and not (len(fstring) > 220 or 'research' in fstring.lower()): #Quick lazy fix from a while ago that got left untouched
            tempmessagef = '###Context:\n' + internet_search(fstring) + '\n'

        messagef = ''
        replystring = ''

        try:
            if (message.reference.resolved.author == client.user and message.reference.resolved != lastmsgd) and not message.reference.resolved.content == lastsent.content:
                if len(message.reference.resolved.content) > 512:
                    inputs = flantokenizer(
                        f'summarize: {message.reference.resolved.content}', return_tensors="pt")
                    outputs = flanmodel.generate(**inputs)
                    replystring = f'Replying to: "{model_name}: {flantokenizer.batch_decode(outputs, skip_special_tokens=True)[0]}" ' # Rarely runs, and result is bad. should change or remove.
                else:
                    if message.reference.resolved.attachments != []:
                        replystring = f'Replying to: "{model_name}: {message.reference.resolved.content} <imggen> image description </imggen>"\n'

                    else:
                        replystring = f'Replying to: "{model_name}: {message.reference.resolved.content}"\n'
                messagef += fstring
            else:
                messagef = fstring
        except AttributeError:
            print("Noreference")
            messagef = fstring

        if imageoutput != "":
            messagef += f' <image>Caption: {imageoutput}; {ocrTranscription}; {foundobj}</image>'

        queue += 1
    except Exception:
        print(traceback.format_exc())

    if checker or messagef == '':
        queue -= 1
        msgqueue[pos + 1] = False
        return
    else:
        try:
            obj.acquire()
        except IndexError:
            msgqueue[pos + 1] = False
        print("passed")

    name = remove_unicode(message.author.name.replace("#0", ''))
    if f'\n{name}:' not in usrlist and f'\n{name}:' != ':':
        usrlist.append(f'\n{name}:')
        print(f'appending {name}: to stop strings.')
        print(usrlist)
    messagef = messagef.replace('?anely', '')
    if re.search(r"<:\w+:\d+>", messagef) or re.search(r"<.:\w+:\d+>", messagef):
        messagef = re.sub(r":\d+>", '', messagef).replace("<a:", '').replace("<:",'')

    prompt = f'{tempmessagef}{name}:\n{messagef}'
    msg = ''
    stiem = 0
    stiem = time.perf_counter()
    gg = ["",""]
    if config.backend == "polymind":
        fmsg = send_message(f'{replystring}{prompt}')
    else:
        gg = infer(f'{replystring}{prompt}',system=contexx.replace('GU9012LD', guild).replace('{daterplc}', f'{today}'),modelname=config.model_name + ":", bsysep=config.llm_parameters['bsysep'],esysep=config.llm_parameters['esysep'],beginsep=config.llm_parameters['beginsep'],endsep=config.llm_parameters['endsep'],mem= history, max_tokens=config.llm_parameters['max_new_tokens'], stopstrings=usrlist, top_p=config.llm_parameters['top_p'], top_k=config.llm_parameters['top_k'], few_shot=f"{config.llm_parameters['beginsep']}itsme9316:\nHello{config.llm_parameters['endsep']}\n{config.llm_parameters['beginsep']}llama:\nHello! How are you?{config.llm_parameters['eos']}{config.llm_parameters['endsep']}\n{config.llm_parameters['beginsep']}itsme9316:\nIm good, please generate an anime style drawing of a woman sitting on the edge of a skyscraper, at daytime,{config.llm_parameters['endsep']}\n{config.llm_parameters['beginsep']}llama:\nSure here you go <imggen>anime style drawing woman sitting on edge of skyscraper, daytime</imggen>{config.llm_parameters['eos']}{config.llm_parameters['endsep']}\n{config.llm_parameters['beginsep']}itsme9316:\nNow get me another image similar to that{config.llm_parameters['endsep']}\n{config.llm_parameters['beginsep']}llama:\nHere you go: <imggen>anime style drawing woman sitting on edge of skyscraper, daytime</imggen>{config.llm_parameters['eos']}{config.llm_parameters['endsep']}")
        fmsg = gg[0]
    history = gg[1]

    odata = fmsg

    tiem = time.perf_counter() - stiem
    print('')
    print(f'Datas done, done in: {tiem}')
    dats = odata

    dats = html.unescape(dats)
    subst = ""
    regex = r"<.*>.*</.*>"
    pattern = "<@(.*)>"
    compiled = re.compile(pattern)

    answr = ""
    #Everything under this is extremely messsy.... need to change
    if ('<imggen>' in dats or ('<editimg>' in dats and raw_image == ''))and enableimg == True and not compiled.search(fstring):
        llamadesc = getllamadesc(dats, messagef)
        try:
            if replystring != '':
                dats += f';j6487{imagegen(llamadesc, guildid, message.reference.resolved.id, False)};j6487'
            else:
                dats += f';j6487{imagegen(llamadesc, guildid, False, False)};j6487'

        except Exception:
            print(traceback.format_exc())
    elif '<editimg>' in dats and enableimg == True and not (raw_image == '') and not compiled.search(fstring):
        llamadesc = getllamadesc(dats, messagef)
        try:
            if replystring != '':
                dats += f';j6487{imagegen(llamadesc, guildid, message.reference.resolved.id, raw_image)};j6487'
            else:
                dats += f';j6487{imagegen(llamadesc, guildid, False, raw_image)};j6487'
        except Exception:
            print(traceback.format_exc())
    elif '<musicgen>' in dats and enableimg == True and not compiled.search(fstring):
        llamadesc = getllamadesc(dats, messagef)
        try:
            dats += musicgen(messagef)
        except Exception:
            print(traceback.format_exc())     
    elif re.match(r".*</.*>", dats) and enableimg == True and not compiled.search(fstring):  
        llamadesc = getllamadesc(dats, messagef)
        try:
            if replystring != '':
                dats += f';j6487{imagegen(llamadesc, guildid, message.reference.resolved.id, False)};j6487'
            else:
                dats += f';j6487{imagegen(llamadesc, guildid, False, False)};j6487'
        except Exception:
            print(traceback.format_exc())      
    elif ("i.imgur" in dats or "imgur.com" in dats or ("<" in dats and "gen" in dats)) and enableimg == True and not compiled.search(fstring):    
        llamadesc = getllamadesc(dats, messagef)
        try:
            if replystring != '':
                dats += f';j6487{imagegen(llamadesc, guildid, message.reference.resolved.id, False)};j6487'
            else:
                dats += f';j6487{imagegen(llamadesc, guildid, False, False)};j6487'     
        except Exception:
            print(traceback.format_exc())     
    elif "<mathjson>" in dats:
        preexp = getllamadesc(dats, messagef, True)
        mask = preexp[1]
        exprs = json.loads(re.sub(r'\s', '',preexp[0]))
        
        solver = create_solver([])
        answr = f'\nAnswer: {solver(exprs)}'
        print(answr)        
        
        dats = dats.replace(mask,'')
    elif re.match(r".*</.*>", dats) and enableimg == False and not compiled.search(fstring):
        subst = "**!!IMAGE GEN IS DISABLED!!**"

    for x in usrlist:
        if '\n' in dats:
            dats = dats.replace(x.replace(':', ''), '')
    if dats == '':
        dats = "*empty response from model.*"
    while globtext != '':
        pass

    else:
        regexm = r";j6487\S*;j6487"
        substm = ""
        mdats = re.sub(regexm, substm, dats, 0, re.MULTILINE)
        mdats = mdats.replace(';jMUSIC62515','')
        dats = re.sub(regex, subst, dats, 0, re.MULTILINE)
        dats = dats.replace('<imggen>', '').replace('<editimg>','')

        msg = [prompt, mdats]
        if '<blockuser>' in dats.lower():
            for x in knownusrs:
                if x[0].lower() in dats.lower():
                    usprbl = x[1].replace("<@", '').replace(">", '')
                    usprbl = re.sub(r"\D", '', usprbl, 0, re.MULTILINE)
                    print(usprbl)
                    blacklist.append(usprbl)
                    print(f'Adding {x[0]} to blacklist')
        for x in knownusrs:
            if x[0].lower() in dats.lower():
                ptrn = re.compile(re.escape(x[0]), re.IGNORECASE)
                dats = ptrn.sub(x[1], dats)
        dats = dats.replace('<blockuser>', '')
        globtext = dats + answr
        messag = message
    if queue > 0:
        queue -= 1
    print("--- %s seconds total ---" % (time.perf_counter() - start_time))
    print("--- %s seconds overhead ---" %
          abs((tiem - (time.perf_counter() - start_time))))
    obj.release()
    msgqueue[pos + 1] = False


async def retardlayer(message, checker, pos ):
    await run(message, checker, pos)

#Old crappy way of multithreading discord.py, probably a better way but overhead is low and so havent looked into replacing it.
def pythonshitconverter(message, checker, pos):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(retardlayer(message, checker, pos))
    loop.close()


@client.event
async def on_ready():
    print('Logged on!')


@client.event
async def on_socket_raw_receive(msg):
    global prev
    global globtext
    global history
    global messag
    global lasttime
    global msgqueue
    global pos
    timef = round(time.time())
    diff = abs(timef - lasttime)
    if diff > 800 and pos != 0:
        msgqueue = [False for i in range(999)]
        pos = 0
        if len(history) > 800:
            history = []
            print("CLEARED")

    if globtext != '': 
        prev = globtext
        await replier(messag, globtext)
        globtext = ''



@client.event
async def on_message(message):
    try:
        guild = message.guild.name
    except AttributeError:
        guild = "PrivateMessage"
    global lastmsg
    global Debugimg
    global lastusr
    global blacklist
    global ignore
    global pos
    global contexx
    global enableimg
    global msgqueue
    global history
    global lastans
    global seed
    global raw
    global lastmsgd
    global sdmem
    global voiceclient
    global config

    if pos > 909:
        msgqueue = [False for i in range(999)]
        pos = 0

    if re.search(r'\bPrivateMessage\b', guild) and config.enabled_features['DMS'] or is_message_allowed(message):
        if message.author == client.user:
            return
        if "!!IMAGE GEN IS DISABLED!!" in message.content and message.author.id != config.adminid: # because some people used to find it funny to spam llama with its own messages.
            return
        if client.user in message.mentions: #Need to change for a cleaner way of doing this.
            command = config.check_command(message.content, message.author.id)
            if not command == None:
                if command['id'] == "clrmem":
                    print("Cleared memory")
                    sdmem = {}
                    contexx = basecontexx
                    history = []
                elif command['id'] == "raw":
                    raw = not raw
                    print(f"Setting sd raw prompt mode to {raw}")
                elif command['id'] == "jvc":
                    try:
                        voiceclient = await message.author.voice.channel.connect()

                    except Exception as e:
                        print(e)
                        print("no voice")
                elif command['id'] == "lvc":
                    try:
                        voiceclient.cleanup()
                        await voiceclient.disconnect()
                        voiceclient = False
                    except Exception as e:
                        print(e)
                        print("no voice")    
                elif command['id'] == "ctxset":
                    contexx = message.content.split("{<")[1]
                    print(f'set context to {contexx}')
                elif command['id'] == "rlmem":
                    with open('memory.json', 'r') as f:
                        history = json.load(f)
                        print(f'Memory length: {len(history)}')
                    print("reloaded mem")
                elif command['id'] == 'lbtmy':
                    del history[-1]
                    del history[-1]
                    print("Cleared last memory")
                elif command['id'] == 'stsed':
                    if seed == -1:
                        seed = 25151
                    else:
                        seed = -1
                    print(f"Set seed to {seed}")
                elif command['id'] == 'eimg':
                    enableimg = True
                    print("Enabled image gen")
                elif command['id'] == 'dimg':
                    enableimg = False
                    print("Disabled image gen")
                elif command['id'] == 'imgdebug':
                    Debugimg = not Debugimg
                    print (f"Image debug set to: {Debugimg}")
                elif command['id'] == 'save':
                    with open('memory.json', 'w') as f:
                        json.dump(history, f)

                    print("Memory saved")
                elif command['id'] == 'block':
                    for u in message.mentions:
                        if u != client.user:
                            blacklist.append(u.id)
                            print(f'{u.name} added to blacklist')
                elif command['id'] == 'clrblock':
                    blacklist = []
                    print("Blacklist cleared")
                elif command['id'] == 'rlcfg':
                    config = Config()
                    print("Config reloaded")
            else:
                try:
                    if lastusr != client.user and lastmsg != message.content: # dumb way of doing this but its been there for a year and it doesnt really hurt anything..
                        pass
                except NameError: #This will always trigger on the first message. Dont remember why but it will.
                    print("bad")
                    lastusr = client.user
                    lastmsg = message.content
                if lastmsg.lower() == message.content.lower() or lastmsg.lower() == lastans.lower():
                    return
                lastusr = message.author
                lastmsg = message.content
                lastmsgd = message

                if message.author.id in blacklist or ignore == True or (len(message.content) > config.llm_parameters["input_message_length_limit"] and message.author.id != 174224265241952256):
                    print("Message blocked")
                else:
                    async with message.channel.typing():
                        checker = False
                        pos += 1
                        print(pos)
                        Thread(target=pythonshitconverter, args=(
                            message, checker, pos,)).start()
# User acc:
client.run(config.token)
