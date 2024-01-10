from google_interface import Google
from os import system
from re import search, sub
import json
import datetime
from gtts import gTTS
import playsound
import openai
import os
import dotenv
import subprocess
import sys

# Load the environment variables
dotenv.load_dotenv()

# get the OpenAI API key, used in packages
API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

# set the openai api key
openai.api_key = API_KEY

# generate directories and log files
# check if directory exists
if not os.path.exists("settings"):
    # if not, create it
    system("mkdir settings")

# if no log file exists, create it
if not os.path.exists("settings/log.txt"):
    with open("settings/log.txt", "w") as f:
        pass

# if no connected apps file exists, create it
if not os.path.exists("settings/connected_apps.json"):
    with open("settings/connected_apps.json", "w") as f:
        # write empty dictionary
        f.write("{}")

# if no settings file exists, create it
if not os.path.exists("settings/settings.json"):
    with open("settings/settings.json", "w") as f:
        # write empty dictionary
        f.write("{}")

# make google search object
google_search = Google()


def speak_message(message, out_loud=True):
    print(message)
    if not out_loud:
        return

    # make text to speech object
    tts = gTTS(
        text=message,
        lang="en",
    )
    tts.save("time.mp3")
    # get current path
    path = os.path.dirname(os.path.abspath(__file__))
    playsound.playsound(f"{path}\\time.mp3")
    os.remove("time.mp3")


def wait_then_parse_dictionary(result, prompt, debug=False):
    # access log file and save prompt and response
    with open("settings/log.txt", "a") as f:
        f.write(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {prompt.encode('unicode_escape').decode('utf-8')}\n{result.encode('unicode_escape').decode('utf-8')}\n"
        )
        f.write("\n")

    # find the dictionary in the response
    # use regex to find the dictionary between { and }
    regex = r"\{[\s\S]*\}"
    dictionary = search(regex, result)
    try:
        # get dictionary as a string
        dictionary = dictionary.group(0)
        # replace double backslashes with single forward slashes
        dictionary = sub(r"\\", "/", dictionary)
        # remove nested quotes
        dictionary = sub(r"\"{", "{", dictionary)
        dictionary = sub(r"}\"", "}", dictionary)
        if debug:
            print("response: ", dictionary)
        # convert to dictionary
        dictionary = json.loads(dictionary)
        # revert quoted none to none and quoted false to false
        for key in dictionary:
            if dictionary[key] == "None":
                dictionary[key] = None
            if dictionary[key] == "False":
                dictionary[key] = False
    except AttributeError:
        dictionary = {
            "action": "error",
            "output": "There was an error parsing the dictionary. Please try again.",
        }
    return dictionary


def handle_request(message, debug=False, browser="www.google.com", speak=False):
    if debug:
        print(f"Asking Jarvis (GPT 3.5 AI Model): {message}")
    speak_message("Just a moment...", out_loud=speak)

    prompt = f"""Normalize the following prompt into a json object with the following keys: action """

    for extension in prompt_extensions:
        prompt += ", " + extension

    prompt += f""", always add a keywords key that is an array. Optional fields:
    response: string (if the user wants to talk to the AI model directly), errorMessage: string (if there is a problem
    with the question). Fields should not be nested in another object. Do not add any extra outputs. Prompt: \"{message}\""""

    result = openai.ChatCompletion.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
        top_p=0,
        max_tokens=1000,
    )

    text = result["choices"][0]["message"]["content"]

    # only print debug info if debug flag is set
    if debug:
        # print the tokens used
        print(f"Tokens used for completion: {result['usage']['completion_tokens']}")
        print(f"Tokens used for prompt: {result['usage']['prompt_tokens']}")

        # print price of prompt and response in usd
        print(f"Total cost in dollars: ${result['usage']['total_tokens'] * 0.000002}")

    dictionary = wait_then_parse_dictionary(text, prompt, debug=debug)
    if dictionary.get("action") == "error":
        print(dictionary.get("output"))
        return

    # get the action
    action = dictionary.get("action")

    if debug:
        print(f"Action: {action}")

    # make settings dictionary
    settings = {
        "debug": debug,
        "out_loud": speak,
        "openai_key": API_KEY,
        "google_search": google_search,
        "browser": browser,
    }

    # open package that corresponds to the action
    try:
        # execute the package and pass the dictionary and settings
        exec(f"""{action}(dictionary, settings)""")
    except Exception as e:
        if "takes 1 positional argument but 2 were given" not in str(e):
            print(f"Error running {action}: {e}")
        else:
            try:
                # execute the package and pass the dictionary
                exec(f"""{action}(dictionary)""")
            except Exception as e:
                print(f"Error running {action}: {e}")


# load app packages using os.walk
requirements = []
package_list = []
for root, dirs, files in os.walk("packages"):
    for app in files:
        if app.endswith(".py") and app.startswith("jarvis_"):
            try:
                # import the app`using the root and app name
                root = root.replace("/", ".").replace("\\", ".")
                exec(
                    f"from {root.replace('/', '.')}.{app[:-3]} import {app[:-3].replace('jarvis_', '')}"
                )
                exec(f"import {root.replace('/', '.')}.{app[:-3]} as {app[:-3]}")
                package_list.append(app[:-3])
            except Exception as e:
                print(f"Error loading {app}: {e}")
        elif app == "jarvis_requirements.txt":
            # add requirements to requirements list
            with open(f"{root}/{app}") as f:
                requirements.extend(f.read().splitlines())
            # remove duplicates from the list
            requirements = list(dict.fromkeys(requirements))

# load all prompt extensions from packages
prompt_extensions = []
for package in package_list:
    try:
        # get the prompt extension from the package
        prompt_extensions.append(eval(f"{package}.prompt_extension"))
    except Exception as e:
        print(f"Error loading prompt extension from {package}: {e}")


# install requirements
if requirements:
    # get installed packages using terminal command
    installed_packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"]
    )
    for package in requirements:
        # check if the package is installed
        if package not in [str(i).split(" ")[0] for i in installed_packages]:
            # install the package using terminal command
            subprocess.check_output([sys.executable, "-m", "pip", "install", package])

# print the list of loaded apps
# print(f"Number of packages loaded: {len(package_list)}: {package_list}")
