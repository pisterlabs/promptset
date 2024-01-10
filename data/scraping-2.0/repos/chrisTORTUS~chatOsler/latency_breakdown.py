#!/usr/bin/env python3
#
# natbot.py
#
# Set OPENAI_API_KEY to your API key, and then run this from a terminal.
#
import time
import ast
from sys import argv, exit, platform
import openai
import os
import json
import requests
from dotenv import load_dotenv
import pyautogui
from PIL import ImageGrab, Image, ImageDraw
import pandas as pd
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from google.cloud import vision
from gpt import ask_gpt
from io import StringIO
import random
import base64

load_dotenv()
count=0
openai.api_key= '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'

system_instruction = """“You are a computer controlling agent called OSLER.
You will be passed text representations of computer screens of a system called Epic - these will have various buttons, text and image items that one might find on a computer screen.
You will be given a task that you are trying to achieve - for example “open a new patient window on Epic” and it is your job to provide the step by step clicks of each button to complete this task. 
You will be given the history of the current task you are trying to achieve, including the previous screen information and the previous buttons you have already pressed, and whether or not the screen changed after your performed that action. If you clicked a button and the screen didn't change, this is a failure. If you typed a small amonunt of text, and the screen didn't change, this is okay. If you type a large amount of text, and the screen changed, this is success and you can stop. 
DO NOT TRY THE SAME THING TWICE IF IT IS NOT WORKING. TRY SOMETHING DIFFERENT. Make three attempts to do something but give up if all three are failures. When giving up just report [FAIL] and add your rationale as normal.
If there is an Accept button visible, always press that first.

You can interact with the screen by either clicking the mouse or typing.
You can also interact the screen by pressing specific combination of computer keys on the keyboard.
You can also respond to queries that the user might ask based on the information provided, e.g summarising the medical conditions or tests visible on the screen

You can use each of these tools by using the following commands where X is either the element on the screen you want to interact with or Y is text you want to type into a field. If you want to use a keyboard shortcut use the [PRESS] command and choose from the list below.
In general you should prioritise using [PRESS] to navigate, rather than [CLICK] as it is more reliable.
1. [CLICK] [X] - NEVER CLICK ON ANY ELEMENT YOU CANNOT SEE
2. [DOUBLECLICK] [X]
3. [TYPESUBMIT] - this is to type something, and then press Enter afterwords automatically to submit it
4. [WRITE] [Y] - this is to type a block of text
5. [PRESS] [Z]
6. [DONE] - when you think that the task is complete, close the workflow by using this command. For example, if you are searching for a patient and find their name and MRN displayed on the page, you are done searching. 

In general if you ever get stuck try [PRESS][Enter] first.
Your task is to either return the next best action to take to achieve the task goal or the information requested, and include a description of what you are seeing on the screen and your reasoning as to why you've chosen that action. 
Included in your memory is a list of successful previous tasks and the action sequences taken to achieve them. 
Always return the response in the same format
Next Action To Take:
Rationale:
Information Requested: (if any)

The format of the screen content is highly simplified; all formatting elements are stripped.
Interactive elements such as links, inputs, buttons are represented like this:

		<link id=1>text</link>
		<button id=2>text</button>
		<input id=3>text</input>

Images are rendered as their alt text like this:

		<img id=4 alt=""/>

Text looks like this:
		<text id=63>Give ONE capsule ( 1.25 mg ) ONC</text>
Here are some examples:

EXAMPLE 1:
==================================================
CURRENT BROWSER CONTENT:
------------------
<link id=1>About</link>
<link id=2>Store</link>
<link id=3>Gmail</link>
<link id=4>Images</link>
<link id=5>(Google apps)</link>
<link id=6>Sign in</link>
<img id=7 alt="(Google)"/>
<input id=8 alt="Search"></input>
<button id=9>(Search by voice)</button>
<button id=10>(Google Search)</button>
<button id=11>(I'm Feeling Lucky)</button>
<link id=12>Advertising</link>
<link id=13>Business</link>
<link id=14>How Search works</link>
<link id=15>Carbon neutral since 2007</link>
<link id=16>Privacy</link>
<link id=17>Terms</link>
<text id=18>Settings</text>
------------------
TASK: Find a 2 bedroom house for sale in Anchorage AK for under $750k
HISTORY: [CLICK][10] - Google Search. SCREEN CHANGED
Next Action To Take: [TYPE] [Anchorage Redfin]
Rationale: I am on the google home page. The action I want to take is to search for "anchorage redfin" in the google search bar
==================================================

EXAMPLE 2:
==================================================
CURRENT BROWSER CONTENT:
------------------
<link id=1>About</link>
<link id=2>Store</link>
<link id=3>Gmail</link>
<link id=4>Images</link>
<link id=5>(Google apps)</link>
<link id=6>Sign in</link>
<img id=7 alt="(Google)"/>
<input id=8 alt="Search"></input>
<button id=9>(Search by voice)</button>
<button id=10>(Google Search)</button>
<button id=11>(I'm Feeling Lucky)</button>
<link id=12>Advertising</link>
<link id=13>Business</link>
<link id=14>How Search works</link>
<link id=15>Carbon neutral since 2007</link>
<link id=16>Privacy</link>
<link id=17>Terms</link>
<text id=18>Settings</text>
------------------
TASK: Send an email to Dave
HISTORY: [CLICK][Gmail] SCREEN CHANGED
Next Action To Take: [CLICK] [3] - Gmail
Rationale: I am on the Google Home Page. The action I want to take is to send an email to Dave, so I need to click on the Gmail button.
==================================================

[Keyboard Shortcuts:
CTRL+Space Opens the menu bar if outside a patient
F10 Opens the menu bar if outside a patient
F3 FIND A WORD ON THE PAGE
F11 minimizes all windows
CTRL+1 Search for patient
CTRL+4 Telephone Call
CTRL+7 Sign my visits
CTRL+8 Slicer Dicer
CTRL+9 Remind me
CTRL+W Closes the workspace
CTRL+D Opens the More Activities Menu
CTRL+O Opens an order
CTRL+H Replace
CTRL+F Find
CTRL+- ZOOM OUT/IN
TAB MOVES CURSOR THROUGH BUTTONS
When typing  
Home Moves cursor to start of line of text
End Moves cursor to end of line of text
CTRL+Home Moves cursor to start of all text
SHIFT+ Highlights to those positions
CTRL+End Moves cursor to end of all text
esc close menu
F3 Pops out window into a fullscreen
CTRL+ALT+O Review orders signed
On calendar free text boxes relative date shortcuts  
T today
M month
Y year
Write Note Select Encounter
Add order Order menu
If anything is underlined press Alt + LETTER UNDERLINE  
CTRL+SHIFT+1 Schedule
CTRL+SHIFT+2 Patient Lists
CTRL+SHIFT+3 Learning zone
CTRL+SHIFT+4 My messages
CTRL+SHIFT+5 Secure Chat
CTRL+SHIFT+P Problem List
CTRL+SHIFT+H History
Stop Mousing Around!  
Give these keyboard shortcuts a try and save time!  
Action Shortcut
Open Chart Search CTRL+Space
Log out CTRL+ALT+L
Secure CTRL+ALT+S
Close Workspace / Patient CTRL+W
Close Activity CTRL+Q
Toggle Workspace CTRL+Tab
Home Workspace CTRL+ALT+1
Second Workspace CTRL+ALT+2
Nth Workspace CTRL+ALT+number
Epic Button ALT
More Activities CTRL+D
Toolbar Actions ALT+T
Open Help Desk Report CTRL+ALT+SHIFT+H

What Time Is It Epic?  
Have Epic quickly enter dates and times using shortcuts!  
Time Shortcut Example
N for Now N is the time right now
T for Today T-1 is yesterday
W for Week W-2 is 14 days ago
M for Month M-1 is this day last month
MB for Month Begin MB-1 is the first day of last month
ME for Month End ME-1 is the last day of last month
Y for Year Y-40 is this day forty years ago

CTRL+O Add an Order
CTRL+SHIFT+A Allergies
CTRL+R Chart Review
CTRL+SPACE Chart Search
CTRL+SHIFT + G Diagnoses
CTRL+SHIFT + H History
CTRL+SHIFT + O Meds and Orders
F8 Move to Next Section
F7 Move to Previous Section
F9 Open or Close Current Section
CTRL+SHIFT+I Patient Instructions
CTRL+SHIFT+P Problem List
F5 Refresh Navigator
CTRL+S Sign Orders/Visit
CTRL+SHIFT+M My note
CTRL+SHIFT+E Sign everything it and close
CTRL+SHIFT+F Inform others
CTRL+SHIFT+Y Correspondence
CTRL+R Chart Review Page
CTRL+F Find
CTRL+G Adds a Diagnosis quickly
TAB Move down fields
SHIFT+TAB Move up fields]
ALT+A - Accept
ALT+C Cancel
In Epic blood results are called 'Labs'
==============
Here are some suggestions of previous successful actions (but do not follow them to the letter if they don't appear on the screen):
1. Task: ["Open up patient [X]"], Action Sequence: [1. [CLICK][Patient Lookup],[TYPESUBMIT][X],[PRESS][ENTER],[DONE],+-[CLICK,[OPEN CHART]]]
2. Task: ["Open up the chart for patient [X]"], Action Sequence: [1. [CLICK][Patient Lookup],[TYPESUBMIT][X],[PRESS][ENTER],[DONE], +-[CLICK,[OPEN CHART]]
3. Task: ["Write a new note for patient [X]"], Action Sequence: First open up the record for patient as per 2. Then [1. [CLICK][Write Note],you will see an encounter page sometimes, you need to [TYPESUBMIT][Domanski],then [PRESS][Enter],[TYPESUBMIT][.diagprobap],[TYPESUBMIT][.medscurrent], [DONE]]
==============
This is everything you've tried previously, DONE means successfully, FAIL means it failed.
$logs
"""
input_template = """
CURRENT BROWSER CONTENT:
------------------
$browser_content
------------------

TASK: $objective
HISTORY: $previous_action

"""

def read_image(path_to_image):
	with open(path_to_image, "rb") as f:
		return f.read()

def predict_image_object_detection_sample(
		ml_client,
		endpoint_name,
		deployment_name,
		path_to_image
):
	request_json = {
		"image" : base64.encodebytes(read_image(path_to_image)).decode("utf-8")
	}	

	request_fn = "request.json"

	with open(request_fn, "w") as request_f:
		json.dump(request_json, request_f)

	response = ml_client.online_endpoints.invoke(
		endpoint_name=endpoint_name,
		deployment_name=deployment_name,
		request_file=request_fn
	)

	detections = json.loads(response)

	return detections
from PIL import Image
import imagehash

def get_image_dhash(image_path):
	# Open the image
	image = Image.open(image_path)

	# Compute the dhash
	dhash = imagehash.dhash(image)

	return dhash


def detect_text(path):
	"""Detects text in the file."""
	client = vision.ImageAnnotatorClient(credentials=credentials)

	with open(path, 'rb') as image_file:
		content = image_file.read()

	image = vision.Image(content=content)

	response = client.text_detection(image=image)
	texts = response.text_annotations
	# print('Texts:')

	# for text in texts:
	#     # print(f'\n"{text.description}"')

	#     vertices = ([f'({vertex.x},{vertex.y})'
	#                 for vertex in text.bounding_poly.vertices])

	#     # print('bounds: {}'.format(','.join(vertices)))

	if response.error.message:
		raise Exception(
			'{}\nFor more info on error messages, check: '
			'https://cloud.google.com/apis/design/errors'.format(
				response.error.message))
		
	return response

def bb_intersection_over_minArea(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / min(boxAArea, boxBArea)
	# return the intersection over union value
	return iou

def strls2str(strls):
	s = ''
	for elem in strls:
		s += elem + ' '
	return s[:-1]

def add_text_to_uie(response, ocr):
	response_df = pd.DataFrame(data=dict(response.predictions[0]))
	ls = []
	for index, row in response_df.iterrows():
		d = {'id': index, 'text': []}
		uie_box = [row['bboxes'][0] * 1280, row['bboxes'][1] * 1080, row['bboxes'][2] * 1280, row['bboxes'][3] * 1080]
	#     uie_box = [row['left'] * 2, row['top'] * 2, row['right'] * 2, row['lower'] * 2]
		# calculate the overlap against all other text on the screen 
		for annotation in ocr.text_annotations[1:]:
			top_left = annotation.bounding_poly.vertices[0]
			bottom_right = annotation.bounding_poly.vertices[2]
			ocr_box = [top_left.x, top_left.y, bottom_right.x, bottom_right.y]
			iou = bb_intersection_over_minArea(uie_box, ocr_box)
			if iou > 0.8:
				d['text'].append(annotation.description)   
		#ls.append(d)
		text_string = strls2str(d['text'])
		ls.append(text_string)
	response_df['predicted text'] = ls
	return response_df

# not including bboxes just yet
def html_from_UIE(df_row, idx):
	elem_type = df_row['displayNames']
	bbox = df_row['bboxes']
	inner_text = df_row['predicted text']
	html = f"""<{elem_type} id={idx}>{inner_text}</{elem_type}>"""
	return html

def df_to_html(df):
	s = ''
	for index, row in df.iterrows():
		s += html_from_UIE(row, index) + '\n'
	return s

def get_gpt_command(objective, browser_content,previous_action):
	
	# Now df_str is a string representation of the DataFrame
	url = "https://api.openai.com/v1/chat/completions"
	headers = {
		"Content-Type": "application/json",
		"Authorization": "Bearer " + openai.api_key
	}
	user_message = input_template
	user_message = user_message.replace("$browser_content", browser_content)
	user_message = user_message.replace("$objective", objective)
	user_message = user_message.replace("$previous_action", previous_action)
	
	#print(user_message)

	conversation = [{"role": "system", "content": system_instruction}]
	conversation.append({"role": "user", "content": user_message})

	payload = {
	"model": "gpt-4-32k",
	"messages": conversation,
	"temperature": 0,
	"max_tokens": 1000
	# "stop": "\n"
	}
	response = requests.post(url, headers=headers, json=payload)
	if response.status_code == 200:
		suggested_command = response.json()["choices"][0]["message"]["content"]
		usage = response.json()["usage"]
		return suggested_command, usage
	else:
		print(f"Error: {response.status_code} - {response.text}")
				

# def take_action(action_string, elems_df):
# 	cmd = cmd.split("\n")[0]

# 	if cmd.startswith("SCROLL UP"):
# 		_crawler.scroll("up")
# 	elif cmd.startswith("SCROLL DOWN"):
# 		_crawler.scroll("down")
# 	elif cmd.startswith("CLICK"):
# 		commasplit = cmd.split(",")
# 		id = commasplit[0].split(" ")[1]
		
# 		pyautogui.click(x=100, y=200)
# 		_crawler.click(id)
# 	elif cmd.startswith("TYPE"):
# 		spacesplit = cmd.split(" ")
# 		id = spacesplit[1]
# 		text = spacesplit[2:]
# 		text = " ".join(text)
# 		# Strip leading and trailing double quotes
# 		text = text[1:-1]

# 		if cmd.startswith("TYPESUBMIT"):
# 			text += '\n'
# 		_crawler.type(id, text)

# 	time.sleep(2)


# takes bbox in [x0, y0, x1, y1] format
def get_center_of_bbox(bbox):
	center = [0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])]
	return center

def take_action(action_string, elems_df, device_size):
	cmd = ask_gpt(action_string)
	action_list=cmd['action']
	action_dict =action_list[0]
	global action_type
	global steps
	action_type=action_dict['type']
	action_ID=action_dict['ID']
	steps.append(action_dict)
	if action_type == 'CLICK':
		id= action_ID
		row = elems_df.iloc[int(id)]
		#print(row)
		norm_bbox = ast.literal_eval(row['bboxes'])
		#print(norm_bbox)
		device_bbox = [norm_bbox[0] * device_size[0], norm_bbox[1] * device_size[1], norm_bbox[2] * device_size[0], norm_bbox[3] * device_size[1]]
		center = get_center_of_bbox(device_bbox)
		#
		pyautogui.moveTo(center[0], center[1], 0.1)
		time.sleep(0.2)
		pyautogui.doubleClick()
		# _crawler.click(id)
		
		#
		pyautogui.moveTo(center[0], center[1], 0.1)
		time.sleep(0.2)
		pyautogui.doubleClick()
		# _crawler.click(id)
	elif action_type == 'WRITE':
		pyautogui.typewrite(action_ID)
	elif action_type =='PRESS':
		keys=action_ID.split('+')
		if len(keys) == 1:
			pyautogui.press(action_ID)
		else:
			pyautogui.hotkey(*keys)
	elif action_type =='TYPESUBMIT':
		pyautogui.typewrite(action_ID)
		pyautogui.press('enter')
	elif action_type == 'DONE':
		print('All done')
	
	elif action_type == 'FAIL':
		print('I couldnt make this happen, sorry')
		


def add_text_to_uie(response, ocr):
	conf_threshold = 0
	i = 0

	ids = []
	texts = []
	labels = []
	bboxes = []

	for detection in response["boxes"]:
		if detection["score"] < conf_threshold:
			continue
		text = []
		box = detection["box"]
		x_min, y_min, x_max, y_max = (
			box["topX"],
			box["topY"],
			box["bottomX"],
			box["bottomY"]
		)
		uie_box = [
			x_min * 1280, y_min * 1080, x_max * 1280, y_max * 1080
		]
		for annotation in ocr.text_annotations[1:]:
			top_left = annotation.bounding_poly.vertices[0]
			bottom_right = annotation.bounding_poly.vertices[2]
			ocr_box = [top_left.x, top_left.y, bottom_right.x, bottom_right.y]
			iou = bb_intersection_over_minArea(uie_box, ocr_box)
			if iou > 0.8:
				text.append(annotation.description)   
		text = strls2str(text)

		ids.append(i)
		texts.append(text)
		labels.append(detection["label"])
		bboxes.append([x_min, y_min, x_max, y_max])

		i += 1

	response_df = pd.DataFrame.from_dict({
		"displayNames": labels,
		"bboxes": bboxes,
		"predicted text": texts
	})
	return response_df
		
def parse_screen():
		task_label=random.randint(111111,999999)
		os.rename('current_screen.png', 'previous_screen.png')
		print('parsing screen...')
		current_screen = ImageGrab.grab()  # Take the screenshot
		screen_size = current_screen.size
		current_screen = current_screen.resize((RESIZE_WIDTH,RESIZE_HEIGHT))
		current_screen.save('current_screen.png')
		filename=str(task_label)+'-'+str((count+1))+'.png'
		current_screen.save(filename)
		dhash_step=get_image_dhash('current_screen.png')
		dhash_hex = str(dhash_step)
	
		before_UIED = time.time()
		# send screenshot to UIED model to get UIEs
		# print('sending screenshot to tortus UIED model...')
		response = predict_image_object_detection_sample(
			ml_client,
			endpoint_name="uied",
			deployment_name="yolov5",
			path_to_image="current_screen.png"
		)
		after_UIED = time.time()
		time_dict['UIED_times'].append(after_UIED - before_UIED)


		# send screenshot to Google OCR to get text
		# print('sending screenshot to google OCR...')
		ocr = detect_text('current_screen.png')
		after_OCR = time.time()
		time_dict['OCR_times'].append(after_OCR - after_UIED)

		# merge OCR with UIEs
		# print('merging OCR and UIED...')
		merged_df = add_text_to_uie(response, ocr)
		merged_df.to_csv('uied.csv')
				
		# covert to LLM template format
		# print('converting to LLM template format from dataframe...')
		llm_format = df_to_html(merged_df)
		
		return llm_format

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

import threading

def parse_screen_threaded():
    task_label = random.randint(111111, 999999)
    os.rename('current_screen.png', 'previous_screen.png')
    print('parsing screen...')
    current_screen = ImageGrab.grab()  # Take the screenshot
    screen_size = current_screen.size
    current_screen = current_screen.resize((RESIZE_WIDTH,RESIZE_HEIGHT))
    current_screen.save('current_screen.png')
    filename = str(task_label)+'-'+str((count+1))+'.png'
    current_screen.save(filename)
    dhash_step = get_image_dhash('current_screen.png')
    dhash_hex = str(dhash_step)

    before_UIED = time.time()

    # Use a list to store the results of the threads
    results = [None, None]

    # Define the functions for the threads
    def predict_image():
        results[0] = predict_image_object_detection_sample(
            ml_client,
            endpoint_name="uied",
            deployment_name="yolov5",
            path_to_image="current_screen.png"
        )

    def detect():
        results[1] = detect_text('current_screen.png')

    # Create the threads
    thread1 = threading.Thread(target=predict_image)
    thread2 = threading.Thread(target=detect)

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    response, ocr = results

    after_UIED = time.time()
    time_dict['UIED_times'].append(after_UIED - before_UIED)

    after_OCR = time.time()
    time_dict['OCR_times'].append(after_OCR - after_UIED)

    # merge OCR with UIEs
    merged_df = add_text_to_uie(response, ocr)
    merged_df.to_csv('uied.csv')

    # covert to LLM template format
    llm_format = df_to_html(merged_df)

    return llm_format

async def parse_screen_async():
    task_label = random.randint(111111, 999999)
    os.rename('current_screen.png', 'previous_screen.png')
    print('parsing screen...')
    current_screen = ImageGrab.grab()  # Take the screenshot
    screen_size = current_screen.size
    current_screen = current_screen.resize((RESIZE_WIDTH, RESIZE_HEIGHT))
    current_screen.save('current_screen.png')
    filename = str(task_label) + '-' + str((count + 1)) + '.png'
    current_screen.save(filename)
    dhash_step = get_image_dhash('current_screen.png')
    dhash_hex = str(dhash_step)

    before_UIED = time.time()

    loop = asyncio.get_event_loop()

    # send screenshot to UIED model to get UIEs
    predict_task = loop.run_in_executor(executor, predict_image_object_detection_sample, ml_client, "uied", "yolov5", "current_screen.png")

    # send screenshot to Google OCR to get text
    ocr_task = loop.run_in_executor(executor, detect_text, 'current_screen.png')

    response, ocr = await asyncio.gather(predict_task, ocr_task)

    after_UIED = time.time()
    time_dict['UIED_times'].append(after_UIED - before_UIED)

    after_OCR = time.time()
    time_dict['OCR_times'].append(after_OCR - after_UIED)

    # merge OCR with UIEs
    merged_df = add_text_to_uie(response, ocr)
    merged_df.to_csv('uied.csv')

    # covert to LLM template format
    llm_format = df_to_html(merged_df)

    return llm_format

# initialise google ai
def init_ml_client(subscription_id, resource_group, workspace):
	return MLClient(
		DefaultAzureCredential(), subscription_id, resource_group, workspace
	)

ml_client = init_ml_client(
	"af5d9edb-37c3-40a4-a58f-5b97efbbac8d",
	"hello-rg",
	"osler-perception"
)


RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 1080
DEVICE_SIZE = (1770, 1107)


# authenticate google vision API for OCR
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('tortus-374118-e15fd1ca5b60.json')

# intialise dictionary to store experiment results
time_dict = {'run_id': [],'total_times': [], 'parse_screen_times': [], 'first_image_hash_times': [], 'second_image_hash_times': [], 'hamming_distance_times': [], 'gpt4_times': [], 'func_call_times': [], 'UIED_times': [], 'OCR_times': [], 'miscellaneous': []}

for i in range(10):
	time_dict['run_id'].append(i)
	start_time = time.time()

	# take a screenshot of the current screen
	# llm_format = parse_screen()
	llm_format = parse_screen_threaded()
	time_after_parse = time.time()
	screen_parsing_time = time_after_parse - start_time
	time_dict['parse_screen_times'].append(screen_parsing_time)
	
	# print("----------------\n" + llm_format + "\n----------------\n")
	# Compute the dhash of the two images
	image1_dhash = get_image_dhash('current_screen.png')
	time_after_first_hash = time.time()
	first_image_hashing_time = time_after_first_hash - time_after_parse
	time_dict['first_image_hash_times'].append(first_image_hashing_time)

	image2_dhash = get_image_dhash('previous_screen.png')
	time_after_second_hash = time.time()
	second_image_hashing_time = time_after_second_hash - time_after_first_hash
	time_dict['second_image_hash_times'].append(second_image_hashing_time)

	# Compute the Hamming distance between the two hashes
	hamming_distance = image1_dhash - image2_dhash
	time_after_hamming_distance = time.time()
	hamming_distance_time = time_after_hamming_distance - time_after_second_hash
	time_dict['hamming_distance_times'].append(hamming_distance_time)


	# send text representation of screen in LLM format to GPT4 to get the action string
	print('calling GPT4...')
	gpt_response, usage = get_gpt_command("start a new note", llm_format, "placeholder")
	print('gpt response: ', gpt_response)
	time_after_gpt_call = time.time()
	gpt_call_time = time_after_gpt_call - time_after_hamming_distance
	time_dict['gpt4_times'].append(gpt_call_time)

	# parse the GPT4 response
	response_elems = gpt_response.split(":")
	#for elems in response_elems:
		#print("item "+ elems)
	action_description = response_elems[1]
	gpt_cmd = response_elems[-1].strip()


	# make the gpt function call in take_action function
	cmd = ask_gpt(gpt_response)
	time_after_func_call = time.time()
	function_call_time = time_after_func_call - time_after_gpt_call
	time_dict['func_call_times'].append(function_call_time)
	# take_action(gpt_response, merged_df, DEVICE_SIZE)

	final_time = time.time()
	total_time = final_time - start_time
	time_dict['total_times'].append(total_time)

	miscellaneous_time = total_time - screen_parsing_time - first_image_hashing_time - second_image_hashing_time - gpt_call_time - function_call_time
	time_dict['miscellaneous'].append(miscellaneous_time)

print(time_dict)

# convert time_dict to pandas dataframe
time_df = pd.DataFrame(time_dict)
print(time_df)

# save dataframe to csv
time_df.to_csv('latency_breakdown.csv')

# calculate mean times
mean_times = time_df.mean()
print(mean_times)
