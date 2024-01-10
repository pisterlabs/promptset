#!/usr/bin/env python3
#
# natbot.py
#
# Set OPENAI_API_KEY to your API key, and then run this from a terminal.
#

from playwright.sync_api import sync_playwright
import time
from sys import argv, exit, platform
import openai
from bardapi import Bard
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import random
from openai_idea import get_idea_from_open_ai

quiet = False
if len(argv) >= 2:
    if argv[1] == '-q' or argv[1] == '--quiet':
        quiet = True
        print(
            "Running in quiet mode (HTML and other content hidden); \n"
            + "exercise caution when running suggested commands."
        )

prompt_template = """

My gmail address is chockynaresh18@gmail.com and password is 8977821555
You are an agent controlling a browser. You are given:

(1) an objective that you are trying to achieve
(2) the URL of your current web page
(3) a simplified text description of what's visible in the browser window (more on that below)

You can issue these commands:
SCROLL UP - scroll up one page
SCROLL DOWN - scroll down one page
CLICK X - click on a given element. You can only click on links, buttons, and inputs!
TYPE X "TEXT" - type the specified text into the input with id X
TYPESUBMIT X "TEXT" - same as TYPE above, except then it presses ENTER to submit the form

The format of the browser content is highly simplified; all formatting elements are stripped.
Interactive elements such as links, inputs, buttons are represented like this:

    <link id=1>text</link>
    <button id=2>text</button>
    <input id=3>text</input>

Images are rendered as their alt text like this:

    <img id=4 alt=""/>

Based on your given objective, issue whatever command you believe will get you closest to achieving your goal.
You always start on Google; you should submit a search query to Google that will take you to the best page for
achieving your objective. And then interact with that page to achieve your objective.


Don't try to interact with elements that you can't see.

Here are some examples:

EXAMPLE 1:
==================================================
CURRENT BROWSER CONTENT:
------------------
<link id=0 aria-label="Gmail (opens a new tab)">Gmail</link>
<link id=1 aria-label="Search for Images (opens a new tab)">Images</link>
<link id=2 aria-label="Google apps"/>
<link id=3>Sign in</link>
<img id=4 Google/>
<button id=5 Search Search/>
<button id=6 Search by voice/>
<button id=7 Search by image/>
<button id=8 Google Search/>
<button id=9 I'm Feeling Lucky/>
<text id=10>Google offered in:</text>
<link id=11>हिन्दी</link>
<link id=12>বাংলা</link>
<link id=13>తెలుగు</link>
<link id=14>मराठी</link>
<link id=15>தமிழ்</link>
<link id=16>ગુજરાતી</link>
<link id=17>ಕನ್ನಡ</link>
<link id=18>മലയാളം</link>
<link id=19>ਪੰਜਾਬੀ</link>
<text id=20>India</text>
<link id=21>About</link>
<link id=22>Advertising</link>
<link id=23>Business</link>
<link id=24>How Search works</link>
<link id=25>Privacy</link>
<link id=26>Terms</link>
<text id=27>Settings</text>


------------------
OBJECTIVE: search for instagram
CURRENT URL: https://www.google.com/
YOUR COMMAND: 
TYPESUBMIT 5 "instagram"

==================================================

EXAMPLE 2:
==================================================
CURRENT BROWSER CONTENT:
------------------
<link id=0 aria-label="Gmail (opens a new tab)">Gmail</link>
<link id=1 aria-label="Search for Images (opens a new tab)">Images</link>
<link id=2 aria-label="Google apps"/>
<link id=3>Sign in</link>
<img id=4 Google/>
<button id=5 Search Search/>
<button id=6 Search by voice/>
<button id=7 Search by image/>
<button id=8 Google Search/>
<button id=9 I'm Feeling Lucky/>
<text id=10>Google offered in:</text>
<link id=11>submit</link>
<link id=12>বাংলা</link>
<link id=13>తెలుగు</link>
<link id=14>मराठी</link>
<link id=15>தமிழ்</link>
<link id=16>ગુજરાતી</link>
<link id=17>ಕನ್ನಡ</link>
<link id=18>മലയാളം</link>
<link id=19>ਪੰਜਾਬੀ</link>
<text id=20>India</text>
<link id=21>About</link>
<link id=22>Advertising</link>
<link id=23>Business</link>
<link id=24>How Search works</link>
<link id=25>Privacy</link>
<link id=26>Terms</link>
<text id=27>Settings</text>


------------------
OBJECTIVE: click the submit button
CURRENT URL: https://www.google.com/
YOUR COMMAND: 
CLICK 11
==================================================

given an objective that you are trying to achieve.
given the URL of the current web page.
given a simplified text description of what's visible in the browser window.
You can issue the following commands:
SCROLL UP - scroll up one page
SCROLL DOWN - scroll down one page
CLICK X - click on a given element. You can only click on links, buttons, and inputs!
TYPE X "TEXT" - type the specified text into the input with id X
TYPESUBMIT X "TEXT" - same as TYPE above, except then it presses ENTER to submit the form
Based on my given objective, you issue whatever command you believe will get me closest to achieving my goal.
you always start on Google; you should submit a search query to Google that will take me to the best page for achieving my objective. And then interact with that page to achieve my objective.


The current browser content, objective, and current URL follow. Reply with your next command to the browser.

CURRENT BROWSER CONTENT:
------------------
$browser_content
------------------

OBJECTIVE: $objective
CURRENT URL: $url
YOUR COMMAND:
"""

black_listed_elements = set(["html", "head", "title", "meta", "iframe", "body", "script", "style", "path", "svg", "br", "::marker",])

class Crawler:
    def __init__(self):
        self.browser = (
            sync_playwright()
            .start()
            .chromium.launch(
                headless=False,
            )
        )
        # options = self.browser.ChromeOptions()
        # options.add_argument("--user-data-dir=C:/Users/{userName}/AppData/Local/Google/Chrome/User Data/Profile {#}/")
        self.page = self.browser.new_page()
        window_size = self.page.viewport_size
        self.page.set_viewport_size({"width": window_size['width'], "height":window_size['height']})

    def go_to_page(self, url):
        self.page.goto(url=url if "://" in url else "http://" + url)
        self.client = self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    # def generate_text(prompt, temperature, best_of, n, max_tokens):
    # 	"""Generates text using Bard.

    # 	Args:
    # 		prompt: The prompt to generate text for.
    # 		temperature: The temperature of the text generation.
    # 		best_of: The number of best text generations to return.
    # 		n: The number of text generations to generate.
    # 		max_tokens: The maximum number of tokens to generate.

    # 	Returns:
    # 		A list of the best text generations.
    # 	"""

    # 	completions = []
    # 	for _ in range(n):
    # 		completion = bard.generate(prompt, temperature, max_tokens)
    # 		completions.append(completion)

    # 	best_completions = random.sample(completions, best_of)
    # 	return best_completions


    def scroll(self, direction):
        if direction == "up":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
            )
        elif direction == "down":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
            )

    def type_in_search_bar(self,query):
        
        search_bar = self.page.wait_for_selector('input[aria-label="Search"]')
        search_bar.fill(query)



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

    def type(self, id, text):
        self.click(id)
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

            if node_name in black_listed_elements:
                continue

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
                ) or node_name == "button" or node_name == 'textarea':
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
            # print("meta", meta_data)
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
        # print("elements_in_view_port",elements_in_view_port)	
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
            # print("meta_data", meta_data)
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
        # print("elements_of_interest", elements_of_interest)
        # elements_to_remove = ["<button id=8 I'm Feeling Lucky/>", '<text id=9>Google offered in:</text>', '<link id=10>हिन्दी</link>', '<link id=11>বাংলা</link>', '<link id=12>తెలుగు</link>', '<link id=13>मराठी</link>', '<link id=14>தமிழ்</link>', '<link id=15>ગુજરાતી</link>', '<link id=16>ಕನ್ನಡ</link>', '<link id=17>മലയാളം</link>', '<link id=18>ਪੰਜਾਬੀ</link>', '<text id=19>India</text>', '<link id=20>About</link>', '<link id=21>Advertising</link>', '<link id=22>Business</link>', '<link id=23>How Search works</link>', '<link id=24>Privacy</link>', '<link id=25>Terms</link>', '<text id=26>Settings</text>']
        # lst = [elem for elem in elements_of_interest if elem not in elements_to_remove]
        return elements_of_interest
        



if (
__name__ == "__main__"
):
    _crawler = Crawler()
    openai.api_key = "Your OPENAI API KEY"

    def print_help():
        print(
            "(g) to visit url\n(u) scroll up\n(d) scroll down\n(c) to click\n(t) to type\n" +
            "(h) to view commands again\n(r/enter) to run suggested command\n(o) change objective"
        )

    def get_gpt_command(objective, url, browser_content):
        prompt = prompt_template
        prompt = prompt.replace("$objective", objective)
        prompt = prompt.replace("$url", url[:100])
        prompt = prompt.replace("$browser_content", browser_content[:4500])
        
        # response = Bard().get_answer(prompt)['content']
        response = openai.Completion.create(model="text-davinci-002",prompt=prompt, temperature=0.7, best_of=10, n=1, max_tokens=64, stop =None)
        #response = Crawler.generate_text(prompt, temperature=0.5, best_of=10, n=3, max_tokens=50)
        print("response",response)
        # return response

        return response.choices[0].text

    def run_cmd(cmd):
        cmd = cmd.split("\n")[0]

        if cmd.startswith("SCROLL UP"):
            _crawler.scroll("up")
        elif cmd.startswith("SCROLL DOWN"):
            _crawler.scroll("down")
        elif cmd.startswith("CLICK"):
            commasplit = cmd.split(",")
            id = commasplit[0].split(" ")[1]
            _crawler.click(id)
        elif cmd.startswith("TYPE"):
            spacesplit = cmd.split(" ")
            id = spacesplit[1]
            text = spacesplit[2:]
            text = " ".join(text)
            # Strip leading and trailing double quotes
            text = text[1:-1]

            if cmd.startswith("TYPESUBMIT"):
                text += '\n'
                print(text)
            _crawler.type(id, text)
            # _crawler.type_in_search_bar(text)
            # _crawler.click(id)




            # Crawler.type_in_search_bar(text)
            # # _crawler.type(id, text)
            # _crawler.click(id)

        time.sleep(2)


    def extract_commands(text):
        commands = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith('SCROLL UP') or line.startswith('SCROLL DOWN') or line.startswith('CLICK') or line.startswith('TYPE') or line.startswith('TYPESUBMIT'):
                command = line.strip()
                commands.append(command)
        commands_str = '\n'.join(commands)
        return commands_str

    def read_objectives_from_file(file_path):
        objectives = []
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                # parts = line.strip().split(' ', 1)
                if len(line) > 1:
                    objective = line
                    objectives.append(objective)		    
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except IOError:
            print("An error occurred while reading the file.")
        print("objectives",objectives)
        return objectives

    task = input("what is your objective??")
    get_idea_from_open_ai(task)
    objective_file_path = "file.txt"
    objectives = read_objectives_from_file(objective_file_path)
    _crawler.go_to_page("https://www.google.com/")


    try:
        for objective in objectives:
            if len(objective) > 0:
                gpt_cmd = ""
                while True:
                    browser_content = "\n".join(_crawler.crawl())
                    prev_cmd = gpt_cmd
                    gpt_cmd = get_gpt_command(objective, _crawler.page.url, browser_content)
                    print("gpt_cmd",gpt_cmd)
                    gpt_cmd = gpt_cmd.strip()
                    print("gptmcnd",gpt_cmd)
                    if not quiet:
                        print("URL: " + _crawler.page.url)
                        print("Objective: " + objective)
                        print("----------------\n" + browser_content + "\n----------------\n")
                    if len(gpt_cmd) > 0:
                        print("lenght0gpt_cmd",gpt_cmd)
                        command = extract_commands(gpt_cmd)
                        print("Suggested command:", command)
                        run_cmd(command)
                        break
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected, exiting gracefully.")
        exit(0)
                        
            

                        
                    


            

