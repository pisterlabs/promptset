from .config import *
from .prompts import *
from .rate_limited_group import *

from googlesearch import search
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import io
import json
import uuid
import datetime
from autogen.agentchat.groupchat import GroupChat
from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant_agent import AssistantAgent
import requests
from urllib.parse import quote

# requires chromadb
# from autogen.agentchat.contrib.teachable_agent import TeachableAgent

import atexit


def get_base_dir():
    if is_running_in_docker():
        return f"memory/{sub_directory}"
    return f"./local_memory/{sub_directory}"


sub_directory = ""


def set_sub_dir(directory):
    global sub_directory, base_directory
    sub_directory = directory
    base_directory = get_base_dir()


base_directory = get_base_dir()


def exit_handler():
    save_log()
    print("Performing cleanup")


# Register the exit_handler function
atexit.register(exit_handler)


def init_user_input(function):
    """
    The function `init_user_input` takes a function as input, prompts the user for a task, and then
    calls the input function with the user's input as an argument.

    :param function: The `function` parameter is a function that will be called with the user input as
    an argument
    """
    start_logging()
    input_message = input("Enter a task: ")
    try:
        function(f"{input_message}")
    except BaseException as e:
        print(f"There was an error {e}")
        # save_log()


chat_log = {}


def start_logging():
    print("Starting Log System")
    autogen.ChatCompletion.start_logging(history_dict=chat_log, compact=True)


def save_log():
    conversation_log = f"""CONVERSATION_LOG:
    {format_json_to_string(autogen.ChatCompletion.logged_history)}
    """
    save_to_file(conversation_log, f"logs/{get_tmp_filename()}.log")
    autogen.ChatCompletion.stop_logging()


func_create_directory = {
    "name": "create_directory",
    "description": f"creates a new directory",
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "directory name to create"},
        },
        "required": ["directory"],
    },
}


def create_directory(directory):
    try:
        directory_path = f"{base_directory}{directory}"
        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        return f"Directory '{directory_path}' created or already exists."
    except Exception as e:
        return f"An error occurred: {e}"


def get_tmp_filename():
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    datestamp = current_datetime.strftime("%Y%m%d%H%M%S")

    # Generate a unique ID
    # unique_id = str(uuid.uuid4())

    # Combine datestamp and unique ID to create a unique identifier
    final_unique_id = f"agent_tmp_{datestamp}"
    return final_unique_id


def format_json_to_string(json_object):
    formatted_string = json.dumps(json_object, indent=2)
    return formatted_string


def post_openai_call(url, payload):
    response = requests.post(
        url=url,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
    )

    if response.status_code != 200:
        raise Exception("Request failed with status code")
    return response


func_text_to_image = {
    "name": "text_to_image",
    "description": "generates an image given an input prompt",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "provide a detailed and specific natural language description that includes the desired objects, their appearance, interactions, contextual information, style or mood preferences, desired actions or poses, any additional specifications or constraints, and ensure it is clear and revised for clarity and specificity, avoid telling it to do something, you are to just describe the image not the act of creating the image.",
            },
            "filename": {
                "type": "string",
                "description": "location to store the returned image",
            },
        },
        "required": ["prompt", "filename"],
    },
}


def text_to_image(prompt, filename):
    # Assuming 'client' is already defined and authenticated elsewhere
    try:
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1,
        }
        response = post_openai_call(url=OAI_IMAGE_GENERATION_URL, payload=payload)

        print(response)
    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An error occurred: {err}"

    # Check if the response is successful and contains the expected data
    if (
        response.status_code == 200
        and "data" in response.json()
        and len(response.json()["data"]) > 0
    ):
        image_url = response.json()["data"][0]["url"]
        print(image_url)
        image_response = requests.get(image_url)
        # Check if the image was successfully downloaded
        if image_response.status_code == 200:
            # Save the image to a file
            file_path = f"{base_directory}{filename}"
            with open(file_path, "wb") as file:
                file.write(image_response.content)
            print(f"Image saved: {file_path}")
            return f"local file path: {file_path}\nurl: {image_url}"
        else:
            print("Failed to download the image.")
    else:
        print("Failed to generate the image.")
    return "FAILED"


func_text_to_speech = {
    "name": "text_to_speech",
    "description": "takes a string of text, and turns it into a spoken version.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "the desired text to be spoken",
            },
            "filename": {
                "type": "string",
                "description": "location to store the file .mp3 file format",
            },
        },
        "required": ["text", "filename"],
    },
}


def text_to_audio(text, filename):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": "onyx",
        },
    )

    if response.status_code != 200:
        raise Exception("Request failed with status code")
    # ...
    # Create an in-memory bytes buffer
    audio_bytes_io = io.BytesIO()

    # Write audio data to the in-memory bytes buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    with open(f"{base_directory}{filename}", "wb") as audio_file:
        audio_file.write(audio_bytes_io.read())

    return f"Stored speech in {base_directory}{filename}"


func_examine_image = {
    "name": "examine_image",
    "description": "Get a detailed summary of what is in an image. Provide a question about what you're looking for, or a generalized idea.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "url of the image to examine",
            },
            "prompt": {
                "type": "string",
                "description": "Provide details about your inquiry into the image.",
            },
        },
        "required": ["url", "prompt"],
    },
}


def examine_image(url, prompt):
    # base64_image = image_to_base64(filename)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                    # "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": api_key,
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content


func_save_to_file = {
    "name": "save_to_file",
    "description": "saves content to the provided filename",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "content to be saved",
            },
            "filename": {
                "type": "string",
                "description": "location to store the file",
            },
        },
        "required": ["content", "filename"],
    },
}


def save_to_file(content, filename):
    file_path = f"{base_directory}{filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"File successfully saved to {file_path}")


func_format_for_markdown = {
    "name": "format_for_markdown",
    "description": "transforms content into a form suitable for markdown files",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "content to be formatted",
            },
        },
        "required": ["content"],
    },
}


def format_for_markdown(content):
    formatted_content = "# Research Report\n\n"  # Markdown header
    formatted_content += content.replace(
        "\n", "\n- "
    )  # Replace newlines with bullet points
    return formatted_content


func_search = {
    "name": "search",
    "description": "google search for relevant information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Google search query",
            },
            "max_results": {
                "type": "integer",
                "description": "(Optional) Maximum number of search results. Default is 10.",
            },
        },
        "required": ["query", "max_results"],
    },
}


# Define research function
def web_search(query, max_results=10):
    # Perform a Google search using the provided query and return a list of URLs
    return list(search(query, num_results=max_results))


func_advanced_search = {
    "name": "advanced_search",
    "description": "arXiv is a free distribution service and an open-access archive for nearly 2.4 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics. Materials on this site are not peer-reviewed by arXiv.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "arXiv search query",
            },
            "max_results": {
                "type": "integer",
                "description": "(Optional) Maximum number of search results. Default is 10.",
            },
        },
        "required": ["query", "max_results"],
    },
}


def search_arXiv(query, max_results=10):
    # Encode the query for URL
    query = quote(query)
    # Construct the arXiv API URL
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code}"


func_scrape = {
    "name": "scrape",
    "description": "Scraping website content based on url",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Website url to scrape",
            },
            "question": {
                "type": "string",
                "description": "(Optional) a specific question about the url",
            },
        },
        "required": ["url"],
    },
}


def scrape(url, question=None):
    """Scrape a website and summarize its content if it's too large."""
    print("Scraping website...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

        # Parse the content of the web page with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        page_text = soup.get_text(separator=" ", strip=True)
        if len(page_text) > 30000:
            summary_text = ""
            if question is not None:
                summary_text = summary(content=page_text, question=question)
            else:
                summary_text = summary(content=page_text)
            return summary_text
        else:
            return page_text
        # return (
        #     soup.prettify()
        # )  # Return the content of the web page formatted as a string
    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An error occurred: {err}"


func_summary = {
    "name": "summary",
    "description": "summarizes some content for you, outputs in markdown",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "content to be summarized",
            },
            "question": {
                "type": "string",
                "description": "(Optional) a specific question about the url. Default: 'Provide a detailed summary, cite references and format in markdown'",
            },
        },
        "required": ["content"],
    },
}


def summary(
    content,
    question="Provide a detailed summary, cite references and format in markdown",
):
    llm = ChatOpenAI(temperature=0, model=api_model)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])

    _summary_template = "{text}"

    map_prompt = f"""
    {question}
    SUMMARY:
    "{_summary_template}"
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(
        input_documents=docs,
    )

    return output


# example not used
def research(query):
    llm_config_researcher = {
        "functions": [
            func_text_to_image,
            func_save_to_file,
            func_format_for_markdown,
            func_text_to_speech,
            func_examine_image,
            func_search,
            func_scrape,
            func_advanced_search,
        ],
        "config_list": config_list,
    }

    research_assistant = autogen.AssistantAgent(
        name="research_assistant",
        system_message=f"""You are a Research Assistant, your goal is to extensivly research the provided topic. Output Desired: a highly detailed outlined report of all technical specifics. Cite all references. At the end of the report, append an empty line and the word "TERMINATE"
        """,
        llm_config=llm_config_researcher,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        code_execution_config={"last_n_messages": 10, "work_dir": f"{get_base_dir()}"},
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "search": web_search,
            "advanced_search": search_arXiv,
            "scrape": scrape,
            "save_to_file": save_to_file,
            "format_for_markdown": format_for_markdown,
            "text_to_speech": text_to_audio,
            "examine_image": examine_image,
            "text_to_image": text_to_image,
        },
    )

    user_proxy.initiate_chat(research_assistant, message=query)

    # Format for markdown (optional step)
    formatted_report = format_for_markdown(user_proxy.last_message()["content"])

    # Save the research report
    save_to_file(formatted_report, "research_report")

    # set the receiver to be researcher, and get a summary of the research report
    user_proxy.stop_reply_at_receive(research_assistant)
    user_proxy.send(
        "Give me the research report that just generated again, return ONLY the report & reference links.",
        research_assistant,
    )

    # return the last message the expert received
    return user_proxy.last_message()["content"]
