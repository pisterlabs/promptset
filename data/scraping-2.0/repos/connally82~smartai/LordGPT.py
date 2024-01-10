

# region ### IMPORTS ###

# Standard Library Imports
import os
import sys
import re
import time
import random
import json
import subprocess
import datetime
import requests.exceptions
from time import sleep
from typing import Dict

# Third-Party Imports
from dotenv import load_dotenv
from termcolor import colored
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from jsmin import jsmin
from fake_useragent import UserAgent
import pdfkit
import requests
import tiktoken

# Local Imports
from scripts.bot_prompts import command_list, bot_prompt, task_prompt
from scripts.bot_commands import botcommands

# Load environment variables
load_dotenv(override=True)
current_path = os.getcwd()
# Working Folder Name
working_folder = os.path.join(current_path, 'LordGPT_folder')
if not os.path.exists(working_folder):
    os.makedirs(working_folder)

# endregion

# region ### MODEL SETTINGS ###
api_function = os.getenv("API_FUNCTION")
max_tokens = int(os.getenv("MAX_TOKENS"))
temperature = float(os.getenv("TEMPERATURE"))
frequency_penalty = float(os.getenv("FREQUENCY_PENALTY"))
presence_penalty = float(os.getenv("PRESENCE_PENALTY"))
top_p = float(os.getenv("TOP_P"))
local_memory_file = os.getenv("LOCAL_MEMORY_FILE", "memory.json")
encoding = tiktoken.get_encoding("cl100k_base")
debug_code = bool(os.getenv("DEBUG_CODE"))
# endregion

# region ### GLOBAL VARIABLES ###

# Message history
message_history = []

#Conversation History Max
#Note:When LordGPT Creates Tasks its saved for life of execution.
max_conversation = 6

# Web, file and Shell content return length
max_characters = 3000

# Debugging settings
# Send requests through BrightData proxy.
proxy_enabled = False

# Global success variable
def set_global_success(value):
    global success
    success = value

# endregion

# region ### FUNCTIONS ###

# Debugging function
def debug_log(message):
    if debug_code:
        print(message)

# Alternate API calls between Azure and OpenAI
api_url = None
api_key = None
model = None
api_type = None
api_count = 0
# API settings, set throttle lower if using alternate to speed up API calls
api_throttle = int(os.environ.get("API_THROTTLE", 10))
api_retry = int(os.environ.get("API_RETRY", 10))
api_timeout = int(os.environ.get("API_TIMEOUT", 60))


def alternate_api(number):
    global api_count
    global api_url
    global api_key
    global max_tokens
    global api_type
    global model

    if api_function == "ALTERNATE":
        api_count = +1
        if number % 2 == 0:
            api_url = os.getenv("AZURE_URL")
            api_key = os.getenv("AZURE_API_KEY")
            model = os.getenv("AZURE_MODEL_NAME")
            api_type = "AZURE"
        else:
            api_url = os.getenv("OPENAI_URL")
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL_NAME")
            api_type = "OPENAI"
    elif api_function == "AZURE":
        api_url = os.getenv("AZURE_URL")
        api_key = os.getenv("AZURE_API_KEY")
        model = os.getenv("AZURE_MODEL_NAME")
        api_type = "AZURE"
    elif api_function == "OPENAI":
        api_url = os.getenv("OPENAI_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL_NAME")
        api_type = "OPENAI"
    else:
        raise ValueError(
            "Invalid API_FUNCTION value. Supported values are 'AZURE', 'OPENAI', or 'ALTERNATE'."
        )
    debug_log(
        "\nAPI Count: "
        + str(api_count)
        + "\nAPI URL: "
        + api_url
        + "\nAPI Key: "
        + api_key
        + "\nAPI Model: "
        + model
        + "\nAPI Type: "
        + api_type
    )
    return api_url, api_key, model, api_type


# Typing effect function
def typing_print(text, color=None):
    if text is None or len(text.strip()) == 0:
        return

    for char in text:
        print(colored(char, color=color) if color else char, end="", flush=True)
        time.sleep(0.005)  # adjust delay time as desired
    print()  # move cursor to the next line after printing the text

def remove_brackets(text):
    return re.sub(r'\[|\]', '', text)

# Print thread function
def print_thread(text):
    typing_print(text)


# Create JSON message function
def create_json_message(
    response_120_words="[DETAILED REASONING]",
    command_string="[COMMAND]",
    command_argument="[ARGUMENT]",
    current_task="[CURRENT TASK]",
    next_task="[NEXT TASK]",
    goal_completion_status="[GOAL COMPLETION %]",
):
    json_message = {
        "response_120_words": response_120_words,
        "command_string": command_string,
        "command_argument": command_argument,
        "current_task": current_task,
        "next_task": next_task,
        "goal_completion_status": goal_completion_status,
    }
    return json.dumps(json_message)


# Get random user agent function
def get_random_user_agent():
    ua = UserAgent()
    browsers = ["Firefox", "Chrome", "Safari", "Opera", "Internet Explorer"]
    operating_systems = ["Windows", "Macintosh", "Linux", "Android", "iOS"]
    browser = random.choice(browsers)
    operating_system = random.choice(operating_systems)

    try:
        user_agent = ua.data_randomize(
            f"{browser} {ua.random}, {operating_system}"
        )  # type: ignore
    except:
        user_agent = ua.random
    return user_agent


# endregion

# region ### API QUERY ###


def query_bot(messages, retries=20):
    alternate_api(api_count)
    time.sleep(api_throttle)
    json_messages = json.dumps(messages)
    for attempt in range(retries):
        try:
            json_payload = json.dumps(
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": presence_penalty,
                    "presence_penalty": presence_penalty,
                }
            )

            headers = {
                "api-key": api_key,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            botresponse = requests.request(
                "POST", api_url, headers=headers, data=json_payload, timeout=45
            )
            debug_log(f"Returned Response from OpenAI: {botresponse.status_code}")
            debug_log(botresponse)
            responseparsed = botresponse.json()["choices"][0]["message"]["content"]
            debug_log(f"Parsed Choices Node: {responseparsed}")
            responseformatted = json.loads(responseparsed)

            if responseformatted is not None:
                if "current_task" in responseformatted:
                    current_task = responseformatted["current_task"]
                    response = responseformatted["response_120_words"]
                    command_string = responseformatted["command_string"]
                    command_argument = responseformatted["command_argument"]
                    next_task = responseformatted["next_task"]
                    goal_status = responseformatted["goal_completion_status"]

                    return (
                        response,
                        command_string,
                        command_argument,
                        current_task,
                        next_task,
                        goal_status,
                    )
                else:
                    alternate_api(api_count)
                    return (
                        "No valid json, ensure you format your responses as the required json",
                        "None",
                        "None",
                        "Reformat Response as json",
                        "Continue where you left off",
                        "Unknown",
                    )
        except Exception as e:
            if attempt < retries - 1:
                print("API Exception...Retrying...")
                alternate_api(api_count)
                time.sleep(2**attempt)
            else:
                raise e


# endregion

# region ### COMMANDS ###

# region ### GENERATE PDF ###


def create_pdf_from_html_markup(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    try:
        # Parse the input string to extract the filename and content
        parts = command_argument.split("Content:")
        filename_part, content = parts[0].strip(), parts[1].strip()
        filename = filename_part.replace("Filename:", "").strip()

        # Concatenate the working_folder path with the filename
        output_path = os.path.join(working_folder, filename)

        # Set up PDFKit configuration (replace the path below with the path to your installed wkhtmltopdf)
        config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")

        if content.lower().endswith('.html'):
            # If content is an HTML file, read the content of the file and pass it to pdfkit
            html_file_path = os.path.join(working_folder, content)
            with open(html_file_path, 'r') as f:
                html_content = f.read()
            pdfkit.from_string(html_content, output_path, configuration=config)
        else:
            # Convert the HTML content to a PDF file using PDFKit
            pdfkit.from_string(content, output_path, configuration=config)

        return create_json_message(
            "PDF Created Successfully",
            command_string,
            command_argument,
            current_task,
            "Determine next task",
            goal_status,
        )
    except Exception as e:
        debug_log(f"Error: {e}")
        return create_json_message(
            "Error: " + str(e),
            command_string,
            command_argument,
            current_task,
            "Research Error",
            goal_status,
        )

# endregion

# region ### SHELL COMMANDS ###

def run_bash_shell_command(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    process = subprocess.Popen(
        command_argument,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=working_folder  # Set the working directory here
    )

    try:
        # Set a timeout value (in seconds) for the command execution
        timeout_value = 120
        output, error = process.communicate(timeout=timeout_value)
    except subprocess.TimeoutExpired:
        process.kill()
        set_global_success(False)
        return create_json_message(
            "Command execution timed out.",
            command_string,
            command_argument,
            "I should research the error",
            next_task,
            goal_status,
        )

    return_code = process.returncode
    debug_log(f"Return Code: {return_code}")

    shell_response = ""

    if "mkdir" in command_argument:
        if return_code == 0:
            set_global_success(True)
            shell_response = "Folder created successfully. " + command_argument
        elif (
            return_code == 1
            and "Folder already exists navigate to folder. " in error.decode("utf-8")
        ):
            set_global_success(True)
            shell_response = (
                "Folder already exists. Try switching to folder. " + command_argument
            )
        else:
            shell_response = f"Error creating folder, research the error: {error.decode('utf-8').strip()}"

    elif "touch" in command_argument:
        if return_code == 0:
            set_global_success(True)
            shell_response = "File created and saved successfully. " + command_argument
        else:
            set_global_success(False)
            shell_response = f"Error creating file, Research the error: {error.decode('utf-8').strip()}"

    else:
        if return_code == 0:
            set_global_success(True)
            # Add slicing to limit output length
            shell_response = (
                "Shell Command Output: "
                + f"{output.decode('utf-8').strip()}"[:max_characters]
            )
        else:
            set_global_success(False)
            # Add slicing to limit error length
            shell_response = f"Shell Command failed, research the error: {error.decode('utf-8').strip()}"[
                :max_characters
            ]

    debug_log(shell_response)
    return create_json_message(
        "BASH Command Output: " + shell_response,
        command_string,
        command_argument,
        "I should analyze the output to ensure success and research any errors",
        next_task,
        goal_status,
    )


# endregion

# region ### ALLOWS MODEL TO CONTINUE ###


def no_command(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    response_string = json.dumps(command_argument)
    set_global_success(True)
    debug_log(f"Response String: {response_string}")
    return create_json_message(
        response, command_string, command_argument, current_task, next_task, goal_status
    )


# endregion

# region ### SAVE RESEARCH ###

def save_research(response, command_string, command_argument, current_task, next_task, goal_status):
    try:
        # Split the command argument into title and content
        title = command_argument.split("Title: ")[1].split(" ResearchContent: ")[0]
        content = command_argument.split("Title: ")[1].split(" ResearchContent: ")[1]
    except IndexError:
        return create_json_message(
            "Error: Invalid format! Please use 'Title: <title> ResearchContent: <content>'.",
            command_string,
            command_argument,
            current_task,
            next_task,
            goal_status,
        )

    # Get the current datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a dictionary with the title, content, and datetime
    research = {"DateTime": current_time, "Title": title, "ResearchContent": content}

    # Save the research to a JSON node
    with open("research.json", "a") as f:
        f.write(json.dumps(research))
        f.write("\n")
    
    return create_json_message(
        "Research saved successfully",
        command_string,
        command_argument,
        current_task,
        next_task,
        goal_status,
    )


# endregion

# region ### FETCH RESEARCH ###


def fetch_research(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    research_list = []
    with open("research.json", "r") as f:
        for line in f:
            research = json.loads(line)
            research_list.append(research)
    formatted_research = ""
    for research in research_list:
        formatted_research += f'DateTime: {research["DateTime"]}\nTitle: {research["Title"]}\nResearchContent: {research["ResearchContent"]}\n\n'
    return create_json_message(
        formatted_research,
        command_string,
        command_argument,
        current_task,
        next_task,
        goal_status,
    )


# endregion

# region ### CREATE TASK LIST ###
# W Writes the task list to bots 2nd message so he always remembers


def create_task_list(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    if command_argument is not None:
        message_handler(None, command_argument, "task")
    return create_json_message(
        "Task List Saved Successfully",
        command_string,
        command_argument,
        current_task,
        next_task,
        goal_status,
    )


# endregion

# region ### CREATE PYTHON SCRIPT ###
# Function names prompt the model


def create_python_script(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    try:
        filename = None
        content = None

        # Extract filename and content using regex
        regex_pattern = r'Filename:\s*(\S+)\s+Content:\s*"""(.*?)"""'
        match = re.search(regex_pattern, command_argument, re.DOTALL)

        if match:
            filename = match.group(1)
            content = match.group(2)
            content = content.replace("\n", "\n")
        else:
            set_global_success(False)
            return create_json_message(
                "Invalid args. Use the Format: Filename: [FILENAME] Content: [CONTENT]",
                command_string,
                command_argument,
                current_task,
                "Research why my command failed",
                goal_status,
            )


        os.makedirs(working_folder, exist_ok=True)
        file_path = os.path.join(working_folder, filename)

        with open(file_path, "w") as file:
            file.write(content)

        set_global_success(True)

        return create_json_message(
            f"Python code created and saved successfully:\nFilename: {filename}\nContent: {command_argument}",
            command_string,
            command_argument,
            current_task,
            next_task,
            goal_status,
        )

    except Exception as e:
        set_global_success(False)
        debug_log(f"Error: {str(e)}")
        return create_json_message(
            f"Error: {str(e)}",
            command_string,
            command_argument,
            "Retry or Reserch Current Task",
            next_task,
            goal_status,
        )


# endregion

# region ### WRITE NEW CONTENT TO FILE ###


def write_new_content_to_file(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    try:
        filename = None
        content = None

        # Extract filename and content using regex
        regex_pattern = r'Filename:\s*(\S+)\s+Content:\s*"""(.*?)"""'
        match = re.search(regex_pattern, command_argument, re.DOTALL)

        if match:
            filename = match.group(1)
            content = match.group(2)
        else:
            set_global_success(False)
            return create_json_message(
                "Invalid args. Use the Format: Filename: [FILENAME] Content: [CONTENT]",
                command_string,
                command_argument,
                current_task,
                "Research why my command failed",
                goal_status,
            )

        if os.path.exists(os.path.join(working_folder, filename)):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"

        file_path = os.path.join(working_folder, filename)

        with open(file_path, "w") as file:
            file.write(content)

        set_global_success(True)
        return create_json_message(
            f"File created and saved successfully\n{content}",
            command_string,
            command_argument,
            current_task,
            next_task,
            goal_status,
        )

    except Exception as e:
        set_global_success(False)
        debug_log(f"Error: {str(e)}")
        return create_json_message(
            f"Error: {str(e)}",
            command_string,
            command_argument,
            "Retry or Reserch Current Task",
            next_task,
            goal_status,
        )


# endregion

# region ## APPEND CONTENT TO FILE ##


def append_content_to_existing_file(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    
    filename, content = command_argument.split(": ", 1)
    file_path = os.path.join(working_folder, filename)

    with open(file_path, "a") as file:
        file.write(content + "\n")

    return create_json_message(
        "File content successfully appended to " + filename,
        command_string,
        command_argument,
        current_task,
        next_task,
        goal_status,
    )


# endregion

# region ### READ CONTENT FROM FILE ###


def read_content_from_file(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    try:
        filename = None
        max_characters = 1000  # define max_characters if not already defined

        args = command_argument.split()

        for i, arg in enumerate(args):
            if arg == "Filename:" and i + 1 < len(args):
                filename = args[i + 1]

        if not filename:
            set_global_success(False)
            debug_log(
                f"Invalid args. {command_argument} Use the Format: Filename: [FILENAME WITH EXT]"
            )
            return create_json_message(
                f"Invalid args. {command_argument} Use the Format: Filename: [FILENAME WITH EXT]",
                command_string,
                command_argument,
                current_task,
                "I will use to proper format and try again",
                goal_status,
            )

        # Concatenate the working_folder path with the filename
        file_path = os.path.join(working_folder, filename)

        if not os.path.exists(file_path):
            set_global_success(False)
            debug_log(
                f"File not found. {command_argument} Use the Format: Filename: [FILENAME WITH EXT]"
            )
            return create_json_message(
                f"File not found. {command_argument} Use the Format: Filename: [FILENAME WITH EXT]",
                command_string,
                command_argument,
                current_task,
                "I will check that my file name is correct or fix the format of my argument",
                goal_status,
            )

        with open(file_path, "r") as file:
            content = file.read()[:max_characters]
        set_global_success(True)
        return create_json_message(
            "File Content: " + f"{content}",
            command_string,
            command_argument,
            current_task,
            next_task,
            goal_status,
        )
    except Exception as e:
        set_global_success(True)
        debug_log(f"Error: {str(e)}")
        return create_json_message(
            "Error: " + f"Error: {str(e)}",
            command_string,
            command_argument,
            current_task,
            "I will research the error",
            goal_status,
        )


# endregion

# region ### SEARCH GOOGLE ###


def search_google(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    try:
        args = command_argument.split("|")

        query = args[0].strip()
        start_index = (
            int(args[1].strip()) if len(args) > 1 and args[1].strip() else None
        )
        num_results = (
            int(args[2].strip()) if len(args) > 2 and args[2].strip() else None
        )
        search_type = args[3].strip() if len(args) > 3 and args[3].strip() else None
        file_type = args[4].strip() if len(args) > 4 and args[4].strip() else None
        site_search = args[5].strip() if len(args) > 5 and args[5].strip() else None
        date_restrict = args[6].strip() if len(args) > 6 and args[6].strip() else None

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": os.environ["GOOGLE_API_KEY"],
            "cx": os.environ["CUSTOM_SEARCH_ENGINE_ID"],
            "q": query,
            "safe": "off",
            "num": 10,
        }

        if num_results:
            params["num"] = min(num_results, 30)

        if start_index:
            params["start"] = start_index

        if search_type:
            params["searchType"] = search_type

        if file_type:
            params["fileType"] = file_type

        if site_search:
            params["siteSearch"] = site_search

        if date_restrict:
            params["dateRestrict"] = date_restrict

        google_response = requests.get(url, params=params)
        data = google_response.json()

        results = []
        if "items" in data:
            for item in data["items"]:
                results.append({"title": item["title"], "link": item["link"]})
        else:
            set_global_success(False)
            return create_json_message(
                "No Search Results Returned",
                command_string,
                command_argument,
                current_task,
                "I will choose another search term",
                goal_status,
            )

        formatted_results = ""
        for result in results:
            formatted_results += f"Google Image Search Results:\n"
            formatted_results += f"Title: {result['title']}\n"
            formatted_results += f"Link: {result['link']}\n\n"
        searchresults = json.dumps(formatted_results)
        debug_log(searchresults)
        set_global_success(True)
        # GOOGLE IMAGE RESULTS RETURNED
        return create_json_message(
            "Search Results: " + searchresults,
            command_string,
            command_argument,
            current_task,
            next_task,
            goal_status,
        )
    except Exception as e:
        debug_log(f"Error: {str(e)}")
        set_global_success(False)
        return create_json_message(
            "No Search Results Returned" + f"Error: {str(e)}",
            command_string,
            command_argument,
            current_task,
            "I will double check my arguments or move to next task.",
            goal_status,
        )


# endregion

# region ### BROWSE WEBSITE ###


def remove_illegal_chars(text):
    return "".join(c for c in text if c.isprintable() and c != "\\")

def scrape_website_url(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    try:
        responsehtml = requests.get(command_argument, timeout=30)
        responsehtml.raise_for_status()
    except requests.RequestException as e:
        return create_json_message(
            "Error: " + f"Error: {str(e)}",
            command_string,
            command_argument,
            current_task,
            next_task,
            goal_status,
        )

    try:
        soup = BeautifulSoup(responsehtml.text, "html.parser")
        content = soup.get_text()
        content = content.replace("\n", " ")
        debug_log(content)
        # Add slicing to limit content length
        content_cleaned = remove_illegal_chars(content)[:max_characters]
        content_compressed = content_cleaned.encode("utf-8")
        content_json_escaped = json.dumps(content_compressed)

    except Exception as e:
        return create_json_message(
            "Error: " + f"Error: {str(e)}",
            command_string,
            command_argument,
            current_task,
            next_task,
            goal_status,
        )

    return create_json_message(
        "Website Content: " + content_json_escaped,
        command_string,
        command_argument,
        current_task,
        next_task,
        goal_status,
    )


# endregion

# region ### MISSION ACCOMPLISHED ###


def mission_accomplished(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    set_global_success(True)
    print("Mission accomplished:", command_argument)
    sys.exit()


# endregion
# endregion

# region ### HANDLER - MESSAGE ###


def message_handler(current_prompt, message, role):
    def update_message_history(role, content):
        try:
            message_history.append({"role": role, "content": content})
        except Exception as e:
            message_history.append(
                {
                    "role": role,
                    "content": "Command did not return anything, let admin know",
                }
            )
            print(
                f"Error occurred while appending message: Check logs, message set to None {e}"
            )

    def limit_message_history():
        while len(message_history) > max_conversation + 1:
            if message_history[3]["role"] != "task":
                message_history.pop(1)
            else:
                message_history.pop(2)

    if len(message_history) == 0:
        message_history.insert(0, {"role": "system", "content": current_prompt})
    elif role == "system":
        message_history[0] = {"role": "system", "content": current_prompt}

    if message is not None:
        if role == "task":
            message_history.insert(1, {"role": "user", "content": message})
            return
        else:
            update_message_history(role, message)

    limit_message_history()
    return message_history


# endregion

# region ### HANDLER - COMMAND ###


def command_handler(
    response, command_string, command_argument, current_task, next_task, goal_status
):
    if not command_string:  # Check if the command_string is empty or None
        return create_json_message(
            "task",
            "command_string Executed Successfully",
            command_string,
            command_argument,
            "Verify Task was executed",
        )

    function = globals().get(command_string)
    if function is None:
        debug_log(
            "Invalid command_string. "
            + command_string
            + " is not a valid command_string."
        )
        return create_json_message(
            "failed",
            "The command_string is invalid, send commands in json format like this: "
            + create_json_message(),
        )
    return function(
        response, command_string, command_argument, current_task, next_task, goal_status
    )


# endregion

# region ### HANDLER - ROUTING ###


def openai_bot_handler(current_prompt, message, role):
    messages = message_handler(current_prompt, message, role)
    (
        response,
        command_string,
        command_argument,
        current_task,
        next_task,
        goal_status,
    ) = query_bot(
        messages
    )  # type: ignore

    print(colored("Overall Goal: ", color="yellow"), end="")
    typing_print(str(goal_status) + "")
    print(colored("LordGPT Thoughts: ", color="green"), end="")
    typing_print(str(response))
    print(colored("Currently : ", color="blue"), end="")
    typing_print(str(current_task) + "")
    print(colored("Next Task: ", color="magenta"), end="")
    typing_print(str(next_task) + "")
    print(colored("Executing Command: ", color="red"), end="")
    typing_print(str(command_string))
    print(colored("Command Argument: ", color="red"), end="")
    typing_print(str(command_argument) + "\n\n")

    handler_response = command_handler(
        response, command_string, command_argument, current_task, next_task, goal_status
    )

    if success == True:
        return handler_response
    return handler_response + "Respond with valid json"


# endregion

# region ### MAIN LOOP ###


def main_loop():
    # Ask the user for the goal of the bot
    print(colored("Goal Tips: ", "green"))
    print(
        colored(
            "1. Tips: Define a clear Goal with steps YOU would take to complete it." +
            "\n2. Example Goal: Find my location, Gather the 5 day weather forecast for my location, save the detailed results of each day to a PDF." +  
            "\n3. The Lord does support more natural language, but does his best when you lay out the steps.." + 
            "\n4. Report any issues to: https://github.com/Cytranics/LordGPT/issues"
            "\n5. If you would like to contribute email agentgrey@thelordg.com", "yellow",
        )
    )

    user_goal = input("Goal: ")
    print(colored("Creating detailed plan to achive the goal....", "green"))
    if not user_goal:
        user_goal = "Find my location, Gather the 5 day weather forecast for my location, save the detailed results of each day to a PDF."
        print(colored("Goal: " + user_goal, "green"))
    set_global_success(True)

    bot_send = openai_bot_handler(bot_prompt + user_goal, "", "assistant")

    while True:
        num_input = input(
            "Enter the amount of responses you want to process automatically (Default 1): "
        )
        try:
            num_iterations = int(num_input) if num_input.strip() else 1
        except ValueError:
            print("Invalid input. Using default value of 1.")
            num_iterations = 1

        for _ in range(num_iterations):
            loop = bot_send
            bot_send = openai_bot_handler(bot_prompt, loop, "assistant")
            loop = bot_send

        continue_choice = input("Is LordG on the right track If not, select n? (y/n): ").lower()
        if continue_choice == "n":
            break


if __name__ == "__main__":
    main_loop()


# endregion
