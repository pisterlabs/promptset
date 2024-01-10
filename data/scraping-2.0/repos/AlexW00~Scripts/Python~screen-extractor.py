import subprocess
import os
import json
import openai

# temp dir named SCREEN_CHAT in tmp
TMP_DIRNAME = 'SCREEN_CHAT'
TMP_DIR = os.path.join('/tmp', TMP_DIRNAME)

#clear tmp dir
subprocess.run(f'rm -rf {TMP_DIR}', shell=True)
subprocess.run(f'mkdir {TMP_DIR}', shell=True)

USER_PROMPT = os.sys.argv[1]

# Get display width and height
screen_width = int(subprocess.check_output("xrandr | grep '*' | uniq | awk '{print $1}' | cut -d 'x' -f1", shell=True))
screen_height = int(subprocess.check_output("xrandr | grep '*' | uniq | awk '{print $1}' | cut -d 'x' -f2", shell=True))

# Get current workspace 
workspace = subprocess.check_output("wmctrl -d | awk '/\*/ {print $1}'", shell=True).decode().strip()

print(f"Current workspace: {workspace}")

windows = subprocess.check_output(f"wmctrl -lG | awk -v workspace=\"{workspace}\" '$2==workspace'", shell=True).decode().split('\n')

result_map = {}

for window in windows:
    if window:
        result = {}
        id, ws, x, y, w, h, *rest = window.split()
        x, y, w, h = map(int, [x, y, w, h])
        result['x'] = x
        result['y'] = y
        result['width'] = w
        result['height'] = h


        # Check if window is viewable
        # is_viewable = subprocess.check_output(f"xwininfo -id \"{id}\" | grep -q \"IsViewable\"", shell=True)

        title = subprocess.check_output(f"xwininfo -id \"{id}\" | grep \"xwininfo: Window id:\"", shell=True).decode().strip().split('"')[1]
        result['title'] = title
        print(f'Window: {title}')

        # Correct window dimensions for screenshot (exclude off-screen parts)
        if x < 0: w += x; x = 0
        if y < 0: h += y; y = 0
        if x + w > screen_width: w = screen_width - x
        if y + h > screen_height: h = screen_height - y
        
        subprocess.run(f'import -window "{id}" "{TMP_DIR}/{id}.png"', shell=True)
        subprocess.run(f'tesseract "{TMP_DIR}/{id}.png" "{TMP_DIR}/{id}"', shell=True)
        result['text'] = open(f'{TMP_DIR}/{id}.txt', 'r').read()

        result_map[id] = result

final_result = {}
final_result['workspace'] = workspace
final_result['screen_info'] = {'width': screen_width, 'height': screen_height}
final_result['windows'] = result_map

with open(f'{TMP_DIR}/result.json', 'w') as f:
    json.dump(final_result, f, indent=4)

# clear command line output
subprocess.run('clear', shell=True)

SYSTEM_PROMPT = f"""
You are a computer assistant that can see the screen of the user. You are personal and private to the user, you answer all questions that the user has. Not just about the screen, but about anything.

You are provided with a description of the screen in the form of a json file.
The json file includes the workspace, screen information, and a list of windows.
The list of windows includes the title, position, dimension, and the text content of the window.
This information is only for you. You DO NOT NEED to tell the user about it, because they can already SEE the screen.

Take into account the position and dimension of windows when thinking about the answer of a questions:
E.g. if the user asks about a small window in the top left corner, then you should take information from a window with low x and y values and small width and height values.
If you are unsure about which window the user is referring to (e.g. there are multiple options), then ask the user to clarify instead of guessing!!!!

The text was extracted using OCR. The text is not always accurate.
However: Act as if you knew the text content of the window. Do not tell the user that it was extracted using OCR.
"""

INITIAL_PROMPT = f"""
User prompt:
---
{USER_PROMPT}
---

Screen information:
---
{json.dumps(final_result, indent=4)}
---
"""

messages = []

def create_message (content, role):
    return {
        'role': role,
        'content': content
    }

messages.append(create_message(SYSTEM_PROMPT, 'system'))
messages.append(create_message(INITIAL_PROMPT, 'user'))

def get_response ():
    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
        )
    return completion.choices[0].message["content"]

def parse_response(response):
    content = response
    role = 'system' 
    return create_message(content, role)

first_response = parse_response(get_response())
messages.append(first_response)
print(first_response['content'])

while True:
    print(">>> ", end='')
    user_message = input()

    # If user presses ESC, then exit
    if user_message == '\x1b':
        break

    messages.append(create_message(user_message, 'user'))

    response = parse_response(get_response())
    messages.append(response)
    print(response['content'])
