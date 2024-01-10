import os
import requests
import json
import base64
import yaml
import mimetypes
import PIL.Image
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

from speech_to_text import speech_to_text

INPUT_CONFIG_PATH = "./src/config.yaml"

# Set Google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
# make model
gemini_model = genai.GenerativeModel('gemini-pro')
gemini_model_vision = genai.GenerativeModel('gemini-pro-vision')

# Set OpenAI API key
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MODEL_VISION = "gpt-4-vision-preview"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def init_config():
    class Inst:
        """This class is used to create objects that can have attributes added dynamically. 
        Without this class, we would not be able to create the config object and its sub-objects 
        (like messages, conversation, etc.) and store the settings loaded from the YAML file."""
        pass

    with open(INPUT_CONFIG_PATH, encoding='utf-8') as data:
        configYaml = yaml.safe_load(data)

    config = Inst()
    # config.messages = Inst()
    # config.messages.loadingModel = configYaml["messages"]["loadingModel"]
    # config.messages.pressSpace = configYaml["messages"]["pressSpace"]
    # config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]

    config.conversation = Inst()
    config.conversation.greeting = configYaml["conversation"]["greeting"]

    config.llm = Inst()
    config.llm.model = configYaml["llm"]["model"]
    config.llm.url = configYaml["llm"]["url"]
    config.llm.stream = configYaml["llm"]["stream"]
    config.llm.systemPrompt = configYaml["llm"]["systemPrompt"]
    config.llm.timeout = configYaml["llm"]["timeout"]

    config.whisperRecognition = Inst()
    config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["model"]

    return config

def gemini(prompt, image_path, chatbot=[]):
    """
    Function to handle gemini model and gemini vision model interactions.
    Parameters:
    prompt (str): The prompt text.
    image_path (str): The path to the image file.
    chatbot (list): A list to keep track of chatbot interactions.
    Returns:
    tuple: Updated chatbot interaction list, an empty string, and None.
    """

    messages = []
    # print(f"{chatbot=}")

    # Process previous chatbot messages if present
    if len(chatbot) != 0:
        for chat in chatbot:
            user, bot = chat[0]['text'], chat[1]['text']
            messages.extend([
                {'role': 'user', 'parts': [user]},
                {'role': 'model', 'parts': [bot]}
            ])
        messages.append({'role': 'user', 'parts': [prompt]})
    else:
        messages.append({'role': 'user', 'parts': [prompt]})

    try:
        # Process image if file is provided
        if image_path is not None:
            with PIL.Image.open(image_path) as img:
                message = [{'role': 'user', 'parts': [prompt, img]}]
                response = gemini_model_vision.generate_content(message)
                gemini_video_resp = response.text
                messages.append({'role': 'model', 'parts': [gemini_video_resp]})

                # Construct list of messages in the required format
                file_data = {
                    "name": os.path.basename(image_path),
                    "path": image_path,
                    "type": mimetypes.guess_type(image_path)[0],
                    "size": os.path.getsize(image_path)
                }
                user_msg = {"text": prompt, "files": [{"file": file_data}]}
                bot_msg = {"text": gemini_video_resp, "files": []}
                chatbot.append([user_msg, bot_msg])
        else:
            response = gemini_model.generate_content(messages)
            gemini_resp = response.text

            # Construct list of messages in the required format
            user_msg = {"text": prompt, "files": []}
            bot_msg = {"text": gemini_resp, "files": []}
            chatbot.append([user_msg, bot_msg])
    except Exception as e:
        # Handling exceptions and raising error to the modal
        print(f"An error occurred: {e}")

    return chatbot

def ollama(prompt, image_path, chatbot=[]):
    messages = []
    # Process previous chatbot messages if present
    if len(chatbot) != 0:
        # print(f"{chatbot=}")
        for chat in chatbot:
            user, bot = chat[0]['text'], chat[1]['text']
            messages.extend([
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': bot}
            ])
        messages.append({'role': 'user', 'content': prompt})
    else:
        messages.append({'role': 'user', 'content': prompt})

    def generate_ollama_response(messages):
        def flatten(lst):
            return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        context = []
        jsonParam = {
            "model": config.llm.model,
            "stream": config.llm.stream,
            "context": context,
            "system": config.llm.systemPrompt,
            "messages": flatten(messages)
        }
        # print(f"{jsonParam=}")
        response = requests.post(
            config.llm.url,
            json=jsonParam,
            headers={'Content-Type': 'application/json'},
            stream=config.llm.stream,
            timeout=config.llm.timeout
        )  # Set the timeout value as per your requirement
        response.raise_for_status()  # raise exception if http calls returned an error
        
        # for non-streaming response
        body = response.json()
        response = body.get('message', '').get('content', '')
        return response

    try:
        # Process image if file is provided
        if image_path is not None:
            message = [{'role': 'user', 'content': prompt, 'images': [image_to_base64(image_path)]}]
            response = generate_ollama_response(message)
            messages.append({'role': 'assistant', 'content': response})

            # Construct list of messages in the required format
            file_data = {
                "name": os.path.basename(image_path),
                "path": image_path,
                "type": mimetypes.guess_type(image_path)[0],
                "size": os.path.getsize(image_path)
            }
            user_msg = {"text": prompt, "files": [{"file": file_data}]}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])
        else:
            response = generate_ollama_response(messages)
            user_msg = {"text": prompt, "files": []}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])

    except Exception as e:
        # Handling exceptions and raising error to the modal
        print(f"An error occurred: {e}")

    return chatbot

def openai(prompt, image_path, chatbot=[]):
    messages = []

    # Process previous chatbot messages if present
    if len(chatbot) != 0:
        # print(f"{chatbot=}")
        for chat in chatbot:
            user, bot = chat[0]['text'], chat[1]['text']
            messages.extend([
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': bot}
            ])
        messages.append({'role': 'user', 'content': prompt})
    else:
        if config.llm.systemPrompt:
            messages.append({'role': 'system', 'content': config.llm.systemPrompt})
        messages.append({'role': 'user', 'content': prompt})
    
    def generate_openai_response(messages):
        def flatten(lst):
            return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        # print(f"{messages=}")
        if isinstance(messages[-1]['content'], list) and 'type' in messages[-1]['content'][-1] and messages[-1]['content'][-1]['type'] == 'image_url':
            model = OPENAI_MODEL_VISION
        else:
            model = OPENAI_MODEL
        # print(f"{model=}")
        # print(f"{messages=}")
        # print(f"{flatten(messages)=}")
        # print("=====================")
        response = openai_client.chat.completions.create(
            model=model,
            messages=flatten(messages),
            max_tokens=300,
            # stream=config.llm.stream
        )
        # print(f"{response=}")

        # for non-streaming response
        response = response.choices[0].message.content
        return response

    try:
        # Process image if file is provided
        if image_path is not None:
            message = [
                {
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image_path}
                    ]
                }
            ]
            response = generate_openai_response(message)
            messages.append({'role': 'assistant', 'content': response})

            user_msg = {"text": prompt, "files": [{"file": image_path}]}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])
        else:
            response = generate_openai_response(messages)
            user_msg = {"text": prompt, "files": []}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])

    except Exception as e:
        # Handling exceptions and raising error to the modal
        print(f"An error occurred: {e}")

    return chatbot

def ask_llm(prompt, image_path=None, chatbot=[]):
    global config
    config = init_config()
    if config.llm.model not in ["gemini", "openai"]:
        # print("Using ollama model")
        chatbot = ollama(prompt, image_path=image_path, chatbot=chatbot)
    elif config.llm.model == "gemini":
        # print("Using gemini model")
        chatbot = gemini(prompt, image_path=image_path, chatbot=chatbot)
    elif config.llm.model == "openai":
        # print("Using openai model")
        chatbot = openai(prompt, image_path=image_path, chatbot=chatbot)  # image_path should be URL
    return chatbot
    
if __name__ == "__main__":
    # chatbot is [[user_msg, bot_msg], [user_msg, bot_msg], ...]

    # paturn 1:  text (example: gemini-pro)
    # print("=====================================")
    # chatbot = ask_llm(prompt="how to output csv from dataframe in python", chatbot=[])
    # user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    # print(f"user: {user}")
    # print(f"bot: {bot}")
    # speech_to_text(text=bot)
    # print("=====================================")
    # chatbot = ask_llm(prompt="how to read it in python", chatbot=chatbot)
    # user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    # print(f"user: {user}")
    # print(f"bot: {bot}")
    # speech_to_text(text=bot)

    # paturn 2:  upload images (example: gemini-pro-vision + gemini-pro)
    print("=====================================")
    chatbot = ask_llm(prompt="What is this image? Discribe the image in detail.", image_path="./temp/temp.jpg")
    user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    print(f"user: {user}")
    print(f"bot: {bot}")
    speech_to_text(text=bot)
    print("=====================================")
    chatbot = ask_llm(prompt="Please describe the image from a different perspective.", chatbot=chatbot)
    user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    print(f"user: {user}")
    print(f"bot: {bot}")
    speech_to_text(text=bot)
