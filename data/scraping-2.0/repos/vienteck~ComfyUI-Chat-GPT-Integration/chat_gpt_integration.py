import importlib
import json
import os
import random
import time
from openai import OpenAI
from datetime import datetime
openAI_models = None
NODE_FILE = os.path.abspath(__file__)

CONFIG_FILE = f"{os.path.dirname(NODE_FILE)}\\config.json"
ROLES_FILE = f"{os.path.dirname(NODE_FILE)}\\roles.json"

def save_prompt_to_file(log):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"prompt_log.txt"), "a") as f:
        f.write(f"Logged {datetime.now().strftime('%Y%m%d%H%M%S')}\n")
        f.write(f"{log}\n\n\n")

def get_roles():
    try:
        # open config file
        roles_list = []
        print(f"Roles File Location: {ROLES_FILE}")
        
        with open(ROLES_FILE, "r") as f:  # Open the file and read the API key
            roles_content = json.load(f)
        for role in roles_content["creativeThinkers"]:
            roles_list.append(role["name"])
    except:
        print("Error: OpenAI API key file not found OpenAI features wont work for you")
        return ""  
    return roles_list

def get_api_key() -> str:
    # Helper function to get the API key from the file
    try:
        # open config file
        print(f"Config File Location: {CONFIG_FILE}")
        with open(CONFIG_FILE, "r") as f:  # Open the file and read the API key
            config = json.load(f)
        api_key = config["openAI_API_Key"]
    except:
        print("Error: OpenAI API key file not found OpenAI features wont work for you")
        return ""
    return api_key  # Return the API key

def get_openAI_models() -> list[str]:
    print('getting ai models')
    global openAI_models
    if openAI_models != None:
        return openAI_models

    install_openai()
    import openai

    # Set the API key for the OpenAI module
    openai.api_key = get_api_key()

    try:
        models = openai.models.list()  # Get the list of models
    except:
        print("Error: OpenAI API key is invalid OpenAI features wont work for you")
        return []

    openAI_models = []  # Create a list for the chat models
    for model in models.data:  # Loop through the models
        openAI_models.append(model.id)  # Add the model to the list

    return openAI_models  # Return the list of chat models

openAI_gpt_models = None

def get_gpt_models() -> list[str]:
    print('getting models')
    global openAI_gpt_models
    if openAI_gpt_models != None:
        return openAI_gpt_models
    models = get_openAI_models()
    openAI_gpt_models = []  # Create a list for the chat models
    for model in models:  # Loop through the models
        if "gpt" in model.lower():
            openAI_gpt_models.append(model)
    return openAI_gpt_models  # Return the list of chat models

def install_openai():
    # Helper function to install the OpenAI module if not already installed
    try:
        importlib.import_module("openai")
    except ImportError:
        import pip

        pip.main(["install", "openai"])
        
def GetPrompt(role, prompt, input_model,append_string) -> str:
    client = OpenAI(
        api_key=get_api_key(),
    )

    if role == "Random":
        role_list= get_roles()
        role = role_list[random.randint(2,len(role_list)-1)]

    print("submitting prompt to OpenAi")
    sb = prompt.strip()
    if len(append_string) > 0 :
        sb += f' - {append_string}'
        
    save_prompt_to_file(sb)
    retryCounter = 0
    while retryCounter < 3:
        try:
            completion = client.chat.completions.create(
                model=input_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a {role} imitator",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            resp = f"Goal: Produce an image in the style of {role}. Content: " +  completion.choices[0].message.content + f' Style: {role}'
            save_prompt_to_file(resp)
            return resp 
        except:
            print('Failed to talk to OpenAi. sleeping for 10 seconds and trying again')
            retryCounter += 1
            time.sleep(10)
        

class ChatGptPrompt:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Multiline string input for the prompt
                "prompt": ("STRING", {"multiline": True}),
                "model": (get_gpt_models(), {"default": "gpt-3.5-turbo-1106"}),
                "role": (get_roles(), {"default": "AI Assistant"})
            },
            "optional": {
                "append_string": ("STRING", {"multiline": True}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "process"

    OUTPUT_NODE = True

    CATEGORY = "OpenAI"  # Define the category for the node
    
    EXECUTE='process'

    @staticmethod
    def process(prompt,model,role,max_words,append_string,seed) -> str:
        text = GetPrompt(role,prompt,model,max_words,append_string)
        print(f'Returning Text: {text}')
        return (text,)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ChatGptPrompt": ChatGptPrompt,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGptPrompt": "ChatGPT Prompt Node",
}
