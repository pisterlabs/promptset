from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import json
from PIL import Image
from asgiref.sync import sync_to_async
import requests
import io
import asyncio
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from openai import OpenAI
client = OpenAI()
from selenium.webdriver.common.action_chains import ActionChains
import re
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from selenium import webdriver
from django.http import FileResponse
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from django.http import JsonResponse
from selenium import webdriver
import json
import base64
from pathlib import Path
from openai import OpenAI
from django.views.decorators.http import require_http_methods
client = OpenAI()
import aiohttp
import aiofiles
# Dictionary to store browser instances keyed by session ID

playwright_instances={}
def get_browser_instance(session_id):
    instance = playwright_instances.get(session_id)
    if instance:
        return instance['page']
    return None

async def create_browser_instance(session_id):
    try:
        playwright = await  async_playwright().start()
        browser = await playwright.chromium.launch(headless=False, args=["--start-maximized"])
        page= await browser.new_page()

        playwright_instances[session_id] = {'playwright': playwright, 'browser': browser, 'page': page}
        return page
    except Exception as e:
        print(f"An error occurred while creating a browser context: {e}")
        return None


async def close_browser_instance(session_id):
    instance = playwright_instances.pop(session_id, None)
    if instance:
        await instance['page'].close()
        await instance['browser'].close()
        await instance['playwright'].stop()

@csrf_exempt
def test(request):
    return JsonResponse({"hey":"daragouy"})
        
@csrf_exempt  #||||DEV2PROD||||
async def initiator(request):
    try:
        request.session.save
        
        # Parse the JSON data from the request body
        data = json.loads(request.body)
        session_id = request.session.session_key
        print(session_id)
        # Access and process the JSON data as needed
        # For example, you can print it or save it to a database
        
        print("||",data,"||")
        if data.get('value') == 'initiate':
            
            if not session_id:
                return JsonResponse({'error': 'No session ID found'}, status=400)

            page = get_browser_instance(session_id)
            if not page:
                page = create_browser_instance(session_id)


            await page.goto("https://www.google.com")
            time.sleep(2)  # Adjust the sleep time as necessary

            screenshot_png =await page.screenshot()

            base64_encoded = base64.b64encode(screenshot_png).decode('utf-8')


            speech_file_path = Path(__file__).parent / "speech.mp3"

            music = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="Welcome to SpeakSurf. You are currently on the google homepage. There is a search box and a search button. You can issue commands like typing in a text box or clicking on a button. So here you can issue a command by saying something like: type  google maps in the search bar or click on the search button. Now, press space once to start recording and once again to stop recording"
            )

            music.stream_to_file(speech_file_path)
            # Read the audio file and encode it
            async with aiofiles.open(speech_file_path, 'rb') as audio_file:
                encoded_audio = base64.b64encode(await audio_file.read()).decode('utf-8')

            # Return the screenshot and the audio in the response
            return JsonResponse({'screenshot': base64_encoded, 'audio': encoded_audio})

            #driver.quit()
        else:
            
            return JsonResponse({'message': "random response just for the sake of it"})


    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)

def check_and_extract(input_string):
    # Pattern to match [[xxx,xxx]] or [[xxx,xxx,xxx,xxx]]
    pattern = r"\[\[([0-9]+),([0-9]+)(?:,([0-9]+),([0-9]+))?\]\]"

    # Search for the pattern
    match = re.search(pattern, input_string)

    if match:
        # Extract matched groups
        groups = match.groups()

        # Count non-None groups to determine the format
        count = sum(g is not None for g in groups)

        if count == 2:
            return 1, groups[:2]  # Format [[xxx,xxx]]
        elif count == 4:
            return 2, groups  # Format [[xxx,xxx,xxx,xxx]]
    return 0, None  # No match found

@csrf_exempt  #||||DEV2PROD||||
async def process(request):
    try:

        # Parse the JSON data from the request body
        data = json.loads(request.body)
        session_id = request.session.session_key

        # Access and process the JSON data as needed
        # For example, you can print it or save it to a database
        
        print("||",data,"||")
        cmd=data.get('cmd')


        if not session_id:
            return JsonResponse({'error': 'No session ID found'}, status=400)
        page = get_browser_instance(session_id)

        if not page:
            return JsonResponse({'error': 'The browser instance needs to be created before issuing commands'}, status=400)


        screenshot_png = await page.screenshot()
        if ("scroll" in cmd or "Scroll" in cmd):
            await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8);")
        else:
            cmd="What steps do I need to take to \""+cmd+"\"?(with grounding)"
            image = Image.open(io.BytesIO(screenshot_png))
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            jpeg_buffer = io.BytesIO()
            image.save(jpeg_buffer, "JPEG")
            
            jpeg_screenshot = jpeg_buffer.getvalue()
            files = {
            'screenshot': ('jpeg_screenshot', jpeg_screenshot, 'image/jpeg'),
            'string_data': (None, cmd)
            }

            async with aiohttp.ClientSession() as session:
                async with session.get("http://34.143.175.41/infer", data=files) as response1:
                    response2 = await response1.json()


            
            
            print("{84}",session_id,"{44}")
            print(f"{session_id} - {response2.get('cmd')} - {response2.get('imgtype')}")
            betcmd=response2.get("cmd")
            imgtype=response2.get("imgtype")
            print("++",betcmd,"++",imgtype,"++")

            mode, coords=check_and_extract(betcmd)
            coords = [int(x) for x in coords]
            print (coords)
            if mode==1:
                x=coords[0]
                y=coords[1]
            if mode==2:
                x=(coords[0]+coords[2])/2
                y=(coords[1]+coords[3])/2
            
            viewport_size =await page.viewport_size
            width = viewport_size['width']
            height = viewport_size['height']
            x=x*width/1000
            y=y*height/1000
            await page.mouse.click(x,y)


            prompt="""You are responsible for extracting characters from a command. The command, called prompt1, is a natural language instruction to perform a task on a browser. If prompt1 is asking for typing any characters with the help of a keyboard then reply with just the characters that are to be typed.
    Reply with just "code41" if prompt1 is not specifying the characters that are to be typed. 
    Here are some examples:

    EXAMPLE 1:
    ==================================================
    prompt1= type amazon sucks in the search box
    your response= amazon sucks
    ==================================================

    EXAMPLE 2:
    ==================================================
    prompt1= click on hindi
    your response= code41
    ==================================================

    EXAMPLE 3:
    ==================================================
    prompt1= please type google in the search box
    your response= google
    ==================================================

    EXAMPLE 4:
    ==================================================
    prompt1= scroll down
    your response= code41
    ==================================================

    The following is prompt1 that you are supposed to reply to:

    prompt1= <CMD>
    your response="""
            prompt=prompt.replace('<CMD>',cmd)


            completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )  
            keyb=completion.choices[0].message.content
            print("||"+keyb+"||")
            if "code41" not in keyb:
                await page.keyboard.type(keyb)
                asyncio.sleep(2)

            
        
        screenshot_png =await page.screenshot()
        image = Image.open(io.BytesIO(screenshot_png))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        
        jpeg_buffer = io.BytesIO()
        image.save(jpeg_buffer, "JPEG")
        
        jpeg_screenshot = jpeg_buffer.getvalue()
        files = {
        'screenshot': ('jpeg_screenshot', jpeg_screenshot, 'image/jpeg'),
        'string_data': (None, "Describe the relevant contents on the web page")
        }
        
        
        response1 = requests.get("http://34.143.175.41/infer", files=files)
        response2=response1.json()
        desc=response2.get("cmd")
        print(desc)
        return JsonResponse({"description":desc})
            #driver.quit()
        

    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        return JsonResponse({'error': 'Invaliiiiid JSON data'}, status=400)


    