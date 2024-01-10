from playwright.sync_api import sync_playwright
import time
from sys import argv, exit, platform
import openai
import os
import json
import requests
import threading
import pyautogui
import pyperclip
from google.cloud import speech
import pyaudio
from six.moves import queue
import re

quiet = False

if len(argv) >= 2:
	if argv[1] == '-q' or argv[1] == '--quiet':
		quiet = True
		print(
			"Running in quiet mode (HTML and other content hidden); \n"
			+ "exercise caution when running suggested commands."
		)

system_instruction = """You are an agent controlling a browser. You are given:

	(1) an objective that you are trying to achieve
	(2) a simplified text description of what's visible in the browser window (more on that below)
	(3) a description of the previous screen and the action you took on it

You can issue these commands:
	SCROLL UP - scroll up one page
	SCROLL DOWN - scroll down one page
	CLICK X - click on a given element. You can only click on links, buttons, and inputs!
	TYPE X "TEXT" - type the specified text into the input with id X
	TYPESUBMIT X "TEXT" - same as TYPE above, except then it presses ENTER to submit the form
	DONE - if you think you have finished the task, simply say "DONE"

The format of the browser content is highly simplified; all formatting elements are stripped.
Interactive elements such as links, inputs, buttons are represented like this:

		<link id=1>text</link>
		<button id=2>text</button>
		<input id=3>text</input>

Images are rendered as their alt text like this:

		<img id=4 alt=""/>

Based on your given objective, current browser content, and description of previous action, think carefully about the next best action to take on which element on the screen to get you closer to completing your objective.
The hard rules you can't break are:
	Don't interact with elements that you can't see
	You have information about the preivous action you took on the previous screen. Never repeat the same action on the same screen
If you think you have finished the task, simply say "DONE".
You need to do two things, one before the other.
First, write a short description of the contents of the screen you see and the specific action you want to take on the specific action on the screen, and how it gets you closer to achieving your objective.
Then, looking at the short description you wrote and the current browser content, suggest one of the above commands that best accomplishes your description of what you want to do.
Your output must always be in the format, with a newline separating your description and your suggested command:
<your description of the screen and action>
<your suggested command>

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
OBJECTIVE: Find a 2 bedroom house for sale in Anchorage AK for under $750k
PREVIOUS ACTIONS:  I am on the youtube home page, and I want to go to the google home page so that I can perform the search for the house


I am on the google home page, and the action I want to take is to search for "anchorage redfin" in the google search bar
TYPESUBMIT 8 "anchorage redfin"
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
OBJECTIVE: Make a reservation for 4 at Dorsia at 8pm
PREVIOUS ACTIONS: I am on the product list page for an amazon search for men's t shirts. I want to go to the google home page.


This is the google home page, and I want to search for "dorsia nyc opentable" in the search bar
TYPESUBMIT 8 "dorsia nyc opentable"
==================================================

EXAMPLE 3:
==================================================
CURRENT BROWSER CONTENT:
------------------
<button id=1>For Businesses</button>
<button id=2>Mobile</button>
<button id=3>Help</button>
<button id=4 alt="Language Picker">EN</button>
<link id=5>OpenTable logo</link>
<button id=6 alt ="search">Search</button>
<text id=7>Find your table for any occasion</text>
<button id=8>(Date selector)</button>
<text id=9>Sep 28, 2022</text>
<text id=10>7:00 PM</text>
<text id=11>2 people</text>
<input id=12 alt="Location, Restaurant, or Cuisine"></input> 
<button id=13>Let’s go</button>
<text id=14>It looks like you're in Peninsula. Not correct?</text> 
<button id=15>Get current location</button>
<button id=16>Next</button>
------------------
OBJECTIVE: Make a reservation for 4 for dinner at Dorsia in New York City at 8pm
PREVIOUS ACTIONS: I am in google docs on a document titled "fundraising". I would like to go to the opentable website.


I am on the opentable home page, and I would like to search for dorsia new york city
TYPESUBMIT 12 "dorsia new york city"
==================================================

The current browser content, objective, and previous action follow. Reply with your description of screen and action, and your suggested command.
Before you reply, make sure you look closely at the description of the previous screen and action, and make sure you DO NOT repeat the same action on the same screen.
"""

input_template = """
CURRENT BROWSER CONTENT:
------------------
$browser_content
------------------

OBJECTIVE: $objective
PREVIOUS ACTIONS: $previous_action
"""

previous_action = ""

letter_gpt_system_msg = """You are a medical office assistant drafting documentation for a physician. Do not add any thoughts or impressions that aren't explicitly mentioned. 
INSTRUCTION: 
Write a professional and jovial letter to the patients GP relaying all of the information in the CONSULTATION TRANSCRIPT. 
DO NOT INCLUDE ANY INFORMATION IN THE LETTER WHICH IS NOT PRESENT IN THE TRANSCRIPT. In the first paragraph include the patients background medical history and current medicines. 
In the second paragraph include the patient's symptoms. In the third paragraph include the family history and social history. 
In the fourth paragraph include the examination findings. In the fifth paragraph include any overall impression/diagnosis made by the physician and include any plans mentioned in the transcript.
Sign the letter as Dr [USER] and add a footnote 'This letter was generated by OSLER, the next generation AI co-pilot for clinicians, and checked by USER' Add a second footnote with a title 'Notes to the Patient' - and add a short explanation written in language for a ten year old of all medical terms and medicines in the text. 
Write each explanation on a separate line. In the text correct any grammatical errors and use generic names for all medications.
"""

soapnote_gpt_system_msg = """You are a medical office assistant drafting documentation for a physician. Do not add any thoughts or impressions that aren't specifically mentioned. 
From the attached transcript generate a S O A P note for the physician to review, include all the relevant information and do not include any information that isn't explicitly mentioned in the transcript. 
If nothing is mentioned just returned [NOT MENTIONED]"""

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
	"""Opens a recording stream as a generator yielding the audio chunks."""

	def __init__(self, rate, chunk):
		self._rate = rate
		self._chunk = chunk

		# Create a thread-safe buffer of audio data
		self._buff = queue.Queue()
		self.closed = True

	def __enter__(self):
		self._audio_interface = pyaudio.PyAudio()
		self._audio_stream = self._audio_interface.open(
			format=pyaudio.paInt16,
			# The API currently only supports 1-channel (mono) audio
			# https://goo.gl/z757pE
			channels=1,
			rate=self._rate,
			input=True,
			frames_per_buffer=self._chunk,
			# Run the audio stream asynchronously to fill the buffer object.
			# This is necessary so that the input device's buffer doesn't
			# overflow while the calling thread makes network requests, etc.
			stream_callback=self._fill_buffer,
		)

		self.closed = False

		return self

	def __exit__(self, type, value, traceback):
		self._audio_stream.stop_stream()
		self._audio_stream.close()
		self.closed = True
		# Signal the generator to terminate so that the client's
		# streaming_recognize method will not block the process termination.
		self._buff.put(None)
		self._audio_interface.terminate()

	def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
		"""Continuously collect data from the audio stream, into the buffer."""
		self._buff.put(in_data)
		return None, pyaudio.paContinue

	def generator(self):
		while not self.closed:
			# Use a blocking get() to ensure there's at least one chunk of
			# data, and stop iteration if the chunk is None, indicating the
			# end of the audio stream.
			chunk = self._buff.get()
			if chunk is None:
				return
			data = [chunk]

			# Now consume whatever other data's still buffered.
			while True:
				try:
					chunk = self._buff.get(block=False)
					if chunk is None:
						return
					data.append(chunk)
				except queue.Empty:
					break

			yield b"".join(data)

class Crawler:
	def __init__(self):
		self.browser = (
			sync_playwright()
			.start()
			.chromium.connect_over_cdp("http://localhost:9222")
		)

		default_context = self.browser.contexts[0]
		self.page = default_context.pages[0]

		self.client = self.page.context.new_cdp_session(self.page)
		self.page.set_viewport_size({"width": 1280, "height": 1080})
		self.page_element_buffer = {}


	def go_to_page(self, url):
		self.page.goto(url=url if "://" in url else "http://" + url)
		self.client = self.page.context.new_cdp_session(self.page)
		self.page_element_buffer = {}

	def scroll(self, direction):
		if direction == "up":
			self.page.evaluate(
				"(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
			)
		elif direction == "down":
			self.page.evaluate(
				"(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
			)

	def click(self, id):
		# Inject javascript into the page which removes the target= attribute from all links
		js = """
		links = document.getElementsByTagName("a");
		for (var i = 0; i < links.length; i++) {
			links[i].removeAttribute("target");
		}
		"""
		self.page.evaluate(js)

		element = self.page_element_buffer.get(int(id))
		if element:
			x = element.get("center_x")
			y = element.get("center_y")
			
			self.page.mouse.click(x, y)
		else:
			print("Could not find element")

	#old type function without deleting existing text in the text field before typing
	# def type(self, id, text):
	# 	self.click(id)
	# 	self.page.keyboard.type(text)

	def type(self, id, text):
		self.click(id)
		self.page.keyboard.press('End')
		self.page.keyboard.press('Shift+Home')
		self.page.keyboard.press('Backspace')
		self.page.keyboard.type(text)

	def enter(self):
		self.page.keyboard.press("Enter")

	def crawl(self):
		page = self.page
		page_element_buffer = self.page_element_buffer
		start = time.time()

		page_state_as_text = []

		device_pixel_ratio = page.evaluate("window.devicePixelRatio")
		if platform == "darwin" and device_pixel_ratio == 1:  # lies
			device_pixel_ratio = 2

		win_scroll_x 		= page.evaluate("window.scrollX")
		win_scroll_y 		= page.evaluate("window.scrollY")
		win_upper_bound 	= page.evaluate("window.pageYOffset")
		win_left_bound 		= page.evaluate("window.pageXOffset") 
		win_width 			= page.evaluate("window.screen.width")
		win_height 			= page.evaluate("window.screen.height")
		win_right_bound 	= win_left_bound + win_width
		win_lower_bound 	= win_upper_bound + win_height
		document_offset_height = page.evaluate("document.body.offsetHeight")
		document_scroll_height = page.evaluate("document.body.scrollHeight")

#		percentage_progress_start = (win_upper_bound / document_scroll_height) * 100
#		percentage_progress_end = (
#			(win_height + win_upper_bound) / document_scroll_height
#		) * 100
		percentage_progress_start = 1
		percentage_progress_end = 2

		page_state_as_text.append(
			{
				"x": 0,
				"y": 0,
				"text": "[scrollbar {:0.2f}-{:0.2f}%]".format(
					round(percentage_progress_start, 2), round(percentage_progress_end)
				),
			}
		)

		tree = self.client.send(
			"DOMSnapshot.captureSnapshot",
			{"computedStyles": [], "includeDOMRects": True, "includePaintOrder": True},
		)

		with open("tree.json", "w") as outfile:
			json.dump(tree, outfile)
		strings	 	= tree["strings"]
		document 	= tree["documents"][0]
		nodes 		= document["nodes"]
		backend_node_id = nodes["backendNodeId"]
		attributes 	= nodes["attributes"]
		node_value 	= nodes["nodeValue"]
		parent 		= nodes["parentIndex"]
		node_types 	= nodes["nodeType"]
		node_names 	= nodes["nodeName"]
		is_clickable = set(nodes["isClickable"]["index"])

		text_value 			= nodes["textValue"]
		text_value_index 	= text_value["index"]
		text_value_values 	= text_value["value"]

		input_value 		= nodes["inputValue"]
		input_value_index 	= input_value["index"]
		input_value_values 	= input_value["value"]

		input_checked 		= nodes["inputChecked"]
		layout 				= document["layout"]
		layout_node_index 	= layout["nodeIndex"]
		bounds 				= layout["bounds"]

		cursor = 0
		html_elements_text = []

		child_nodes = {}
		elements_in_view_port = []

		anchor_ancestry = {"-1": (False, None)}
		button_ancestry = {"-1": (False, None)}

		def convert_name(node_name, has_click_handler):
			if node_name == "a":
				return "link"
			if node_name == "input":
				return "input"
			if node_name == "img":
				return "img"
			if (
				node_name == "button" or has_click_handler
			):  # found pages that needed this quirk
				return "button"
			else:
				return "text"

		def find_attributes(attributes, keys):
			values = {}

			for [key_index, value_index] in zip(*(iter(attributes),) * 2):
				if value_index < 0:
					continue
				key = strings[key_index]
				value = strings[value_index]

				if key in keys:
					values[key] = value
					keys.remove(key)

					if not keys:
						return values

			return values

		def add_to_hash_tree(hash_tree, tag, node_id, node_name, parent_id):
			parent_id_str = str(parent_id)
			if not parent_id_str in hash_tree:
				parent_name = strings[node_names[parent_id]].lower()
				grand_parent_id = parent[parent_id]

				add_to_hash_tree(
					hash_tree, tag, parent_id, parent_name, grand_parent_id
				)

			is_parent_desc_anchor, anchor_id = hash_tree[parent_id_str]

			# even if the anchor is nested in another anchor, we set the "root" for all descendants to be ::Self
			if node_name == tag:
				value = (True, node_id)
			elif (
				is_parent_desc_anchor
			):  # reuse the parent's anchor_id (which could be much higher in the tree)
				value = (True, anchor_id)
			else:
				value = (
					False,
					None,
				)  # not a descendant of an anchor, most likely it will become text, an interactive element or discarded

			hash_tree[str(node_id)] = value

			return value

		for index, node_name_index in enumerate(node_names):

			node_parent = parent[index]
			node_name = strings[node_name_index].lower()

			is_ancestor_of_anchor, anchor_id = add_to_hash_tree(
				anchor_ancestry, "a", index, node_name, node_parent
			)

			is_ancestor_of_button, button_id = add_to_hash_tree(
				button_ancestry, "button", index, node_name, node_parent
			)

			try:
				cursor = layout_node_index.index(
					index
				)  # todo replace this with proper cursoring, ignoring the fact this is O(n^2) for the moment
			except:
				continue

			# if node_name in black_listed_elements:
			# 	continue

			[x, y, width, height] = bounds[cursor]
			x /= device_pixel_ratio
			y /= device_pixel_ratio
			width /= device_pixel_ratio
			height /= device_pixel_ratio

			elem_left_bound = x
			elem_top_bound = y
			elem_right_bound = x + width
			elem_lower_bound = y + height

			partially_is_in_viewport = (
				elem_left_bound < win_right_bound
				and elem_right_bound >= win_left_bound
				and elem_top_bound < win_lower_bound
				and elem_lower_bound >= win_upper_bound
			)

			if not partially_is_in_viewport:
				continue

			meta_data = []

			# inefficient to grab the same set of keys for kinds of objects but its fine for now
			element_attributes = find_attributes(
				attributes[index], ["type", "placeholder", "aria-label", "title", "alt"]
			)

			ancestor_exception = is_ancestor_of_anchor or is_ancestor_of_button
			ancestor_node_key = (
				None
				if not ancestor_exception
				else str(anchor_id)
				if is_ancestor_of_anchor
				else str(button_id)
			)
			ancestor_node = (
				None
				if not ancestor_exception
				else child_nodes.setdefault(str(ancestor_node_key), [])
			)

			if node_name == "#text" and ancestor_exception:
				text = strings[node_value[index]]
				if text == "|" or text == "•":
					continue
				ancestor_node.append({
					"type": "type", "value": text
				})
			else:
				if (
					node_name == "input" and element_attributes.get("type") == "submit"
				) or node_name == "button":
					node_name = "button"
					element_attributes.pop(
						"type", None
					)  # prevent [button ... (button)..]
				
				for key in element_attributes:
					if ancestor_exception:
						ancestor_node.append({
							"type": "attribute",
							"key":  key,
							"value": element_attributes[key]
						})
					else:
						meta_data.append(element_attributes[key])

			element_node_value = None

			if node_value[index] >= 0:
				element_node_value = strings[node_value[index]]
				if element_node_value == "|": #commonly used as a seperator, does not add much context - lets save ourselves some token space
					continue
			elif (
				node_name == "input"
				and index in input_value_index
				and element_node_value is None
			):
				node_input_text_index = input_value_index.index(index)
				text_index = input_value_values[node_input_text_index]
				if node_input_text_index >= 0 and text_index >= 0:
					element_node_value = strings[text_index]

			# remove redudant elements
			if ancestor_exception and (node_name != "a" and node_name != "button"):
				continue

			elements_in_view_port.append(
				{
					"node_index": str(index),
					"backend_node_id": backend_node_id[index],
					"node_name": node_name,
					"node_value": element_node_value,
					"node_meta": meta_data,
					"is_clickable": index in is_clickable,
					"origin_x": int(x),
					"origin_y": int(y),
					"center_x": int(x + (width / 2)),
					"center_y": int(y + (height / 2)),
				}
			)
		# print("\n\n========elements in view_port=============\n\n")
		# print(elements_in_view_port)

		# lets filter further to remove anything that does not hold any text nor has click handlers + merge text from leaf#text nodes with the parent
		elements_of_interest= []
		id_counter 			= 0

		for element in elements_in_view_port:
			node_index = element.get("node_index")
			node_name = element.get("node_name")
			node_value = element.get("node_value")
			is_clickable = element.get("is_clickable")
			origin_x = element.get("origin_x")
			origin_y = element.get("origin_y")
			center_x = element.get("center_x")
			center_y = element.get("center_y")
			meta_data = element.get("node_meta")

			inner_text = f"{node_value} " if node_value else ""
			meta = ""
			
			if node_index in child_nodes:
				for child in child_nodes.get(node_index):
					entry_type = child.get('type')
					entry_value= child.get('value')

					if entry_type == "attribute":
						entry_key = child.get('key')
						meta_data.append(f'{entry_key}="{entry_value}"')
					else:
						inner_text += f"{entry_value} "

			if meta_data:
				meta_string = " ".join(meta_data)
				meta = f" {meta_string}"

			if inner_text != "":
				inner_text = f"{inner_text.strip()}"

			converted_node_name = convert_name(node_name, is_clickable)

			# not very elegant, more like a placeholder
			if (
				(converted_node_name != "button" or meta == "")
				and converted_node_name != "link"
				and converted_node_name != "input"
				and converted_node_name != "img"
				and converted_node_name != "textarea"
			) and inner_text.strip() == "":
				continue

			page_element_buffer[id_counter] = element

			if inner_text != "": 
				elements_of_interest.append(
					f"""<{converted_node_name} id={id_counter}{meta}>{inner_text}</{converted_node_name}>"""
				)
			else:
				elements_of_interest.append(
					f"""<{converted_node_name} id={id_counter}{meta}/>"""
				)
			id_counter += 1

		print("Parsing time: {:0.2f} seconds".format(time.time() - start))
		return elements_of_interest
	

class Actor(threading.Thread):
	def __init__(self,stop_event) -> None:
		threading.Thread.__init__(self)
		self.crawler = Crawler()
		self.stop_event = stop_event
		self.consultation_transcript = ""
	
	def get_gpt_response(self, objective, browser_content):
		url = "https://api.openai.com/v1/chat/completions"
		headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " + openai.api_key
		}
		user_message = input_template
		user_message = user_message.replace("$browser_content", browser_content)
		user_message = user_message.replace("$objective", objective)
		user_message = user_message.replace("$previous_action", previous_action)

		conversation = [{"role": "system", "content": system_instruction}]
		conversation.append({"role": "user", "content": user_message})

		payload = {
		"model": "gpt-4",
		"messages": conversation,
		"temperature": 0,
		"max_tokens": 200
		# "stop": "\n"
		}
		response = requests.post(url, headers=headers, json=payload)
		if response.status_code == 200:
			suggested_command = response.json()["choices"][0]["message"]["content"]
			return suggested_command
		else:
			print(f"Error: {response.status_code} - {response.text}")

	def run_cmd(self, cmd):
		cmd = cmd.split("\n")[0]
		
		if cmd.startswith("SCROLL UP"):
			self.crawler.scroll("up")
		elif cmd.startswith("SCROLL DOWN"):
			self.crawler.scroll("down")
		elif cmd.startswith("CLICK"):
			commasplit = cmd.split(",")
			id = commasplit[0].split(" ")[1]
			self.crawler.click(id)
		elif cmd.startswith("TYPE"):
			spacesplit = cmd.split(" ")
			id = spacesplit[1]
			text = spacesplit[2:]
			text = " ".join(text)
			# Strip leading and trailing double quotes
			text = text[1:-1]

			if cmd.startswith("TYPESUBMIT"):
				text += '\n'
			self.crawler.type(id, text)

		time.sleep(2)

	def perform_command(self, objective):

		gpt_cmd = ""
		prev_cmd = ""

		try:
			while not self.stop_event.is_set():
				# scrape the page
				browser_content = "\n".join(self.crawler.crawl())

				# send objective and page contents to GPT and get back response
				gpt_response = self.get_gpt_response(objective, browser_content)

				# parse the response
				response_elems = gpt_response.split("\n")
				action_description = response_elems[0]
				gpt_cmd = response_elems[-1].strip()

				# printing and logging
				print("URL: " + self.crawler.page.url)
				print("----------------\n" + browser_content + "\n----------------\n")
				print("Objective: " + objective)
				print(str(gpt_response))
				# log.write("URL: " + self.crawler.page.url)
				# log.write("\n----------------\n" + browser_content + "\n----------------\n")
				# log.write("\nObjective: " + objective)
				# log.write("\n" + str(gpt_response))

				if not quiet:
					print("URL: " + self.crawler.page.url)
					print("Objective: " + objective)
					print("----------------\n" + browser_content + "\n----------------\n")
					# log.write("----------------\n" + browser_content + "\n----------------\n")
				if len(gpt_cmd) > 0:
					print('Description of action: ' + gpt_response)
					print("Suggested command: " + gpt_cmd)
					# log.write('Description of action: ' + gpt_response + '\n')
					# log.write("Suggested command: " + gpt_cmd + '\n')

				self.run_cmd(gpt_cmd)

		except KeyboardInterrupt:
			print("\n[!] Ctrl+C detected, exiting gracefully.")
			exit(0)

	def listen_print_loop(self, responses):
		"""Iterates through server responses and prints them.

		The responses passed is a generator that will block until a response
		is provided by the server.

		Each response may contain multiple results, and each result may contain
		multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
		print only the transcription for the top alternative of the top result.

		In this case, responses are provided for interim results as well. If the
		response is an interim one, print a line feed at the end of it, to allow
		the next result to overwrite it, until the response is a final one. For the
		final one, print a newline to preserve the finalized transcription.
		"""
		num_chars_printed = 0
		for response in responses:
			if not response.results:
				continue

			# The `results` list is consecutive. For streaming, we only care about
			# the first result being considered, since once it's `is_final`, it
			# moves on to considering the next utterance.
			result = response.results[0]
			if not result.alternatives:
				continue

			# Display the transcription of the top alternative.
			transcript = result.alternatives[0].transcript

			# Display interim results, but with a carriage return at the end of the
			# line, so subsequent lines will overwrite them.
			#
			# If the previous result was longer than this one, we need to print
			# some extra spaces to overwrite the previous result
			overwrite_chars = " " * (num_chars_printed - len(transcript))

			if not result.is_final:
				# sys.stdout.write(transcript + overwrite_chars + "\r")
				# sys.stdout.flush()

				# num_chars_printed = len(transcript)
				pass

			else:
				
				# print(transcript + overwrite_chars)
				output = transcript + overwrite_chars
				self.consultation_transcript += output
				output = output.lower()

				if "stop recording" in output:
					break

				print(output)
				pyperclip.copy(transcript + overwrite_chars)
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')

				# Exit recognition if any of the transcribed phrases could be
				# one of our keywords.
				if re.search(r"\b(exit|quit)\b", transcript, re.I):
					print("Exiting..")
					break

				num_chars_printed = 0


	def transcribe_consultation(self):
		# add header
		pyperclip.copy('\n\n--------- Consultation Transcription ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		# See http://g.co/cloud/speech/docs/languagesv
		# for a list of supported languages.
		language_code = "en-US"  # a BCP-47 language tag

		client = speech.SpeechClient()
		config = speech.RecognitionConfig(
			encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
			sample_rate_hertz=RATE,
			language_code=language_code,
			model='default'
		)

		streaming_config = speech.StreamingRecognitionConfig(
			config=config, interim_results=True
		)

		with MicrophoneStream(RATE, CHUNK) as stream:
			audio_generator = stream.generator()
			requests = (
				speech.StreamingRecognizeRequest(audio_content=content)
				for content in audio_generator
			)

			responses = client.streaming_recognize(streaming_config, requests)

			# Now, put the transcription responses to use.
			self.listen_print_loop(responses)

	def summarise_transcription(self):
		# add header
		pyperclip.copy('\n\n--------- Consultation Summary ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		#make call to GPT-4 to create SOAP note from consultation transcript
		url = "https://api.openai.com/v1/chat/completions"
		headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " + openai.api_key
		}

		soapnote_gpt_user_msg = "CONSULTATION TRANSCRIPT:\n" + self.consultation_transcript + "\n"
		conversation = [{"role": "system", "content": soapnote_gpt_system_msg}]
		conversation.append({"role": "user", "content": soapnote_gpt_user_msg})

		payload = {
		"model": "gpt-4",
		"messages": conversation,
		"temperature": 0,
		"max_tokens": 500
		# "stop": "\n"
		}
		response = requests.post(url, headers=headers, json=payload)
		if response.status_code == 200:
			soap_note = response.json()["choices"][0]["message"]["content"]
		else:
			print(f"Error: {response.status_code} - {response.text}")


		self.soapnote = soap_note

		# write consultation summary to notes
		pyperclip.copy(soap_note)
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')
