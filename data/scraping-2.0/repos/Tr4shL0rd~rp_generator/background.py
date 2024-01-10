"""dwa"""
import os
import os.path
import openai
import requests
import websocket
import json
import itertools
import sys
import base64
import uuid
from dotenv import load_dotenv
from helper import IMAGE_PATH
from helper import Character
from helper import get_current_image_model
load_dotenv()
APIKEY = os.environ["APIKEY"]
openai.api_key = APIKEY

models = {
    "davinci": "davinci",
    "davinci-003": "text-davinci-003",
}
BASE_KEYWORDS = [
                "\"World of Warcraft\"",
                "Warcraft",
                "Azeroth",
                "Fantasy",
                "Character Creation",
                "Creation",
                "Character Backstory",
                "Character",
                "Backstory",
                "Story",
                "DND",
                "\"Dungeons and Dragons\"",
                "\"Dungeons & Dragons\""
                ]

def download_image(url:str, character:Character):
    """downloads an image from URL"""
    image_data = requests.get(url, timeout=3).content
    character_name = character.Name.replace(" ", "-")
    with open(f"{os.path.join(IMAGE_PATH,character_name)}.png", "wb") as image_file:
        image_file.write(image_data)
    print(f"\rsaved to {os.path.join(IMAGE_PATH, character.Name.replace(' ','-'))}.png")


def get_keywords(character:Character):
    """
    appends the generated character paramentes to the keywords
    """
    return BASE_KEYWORDS + [character.Presenting_gender,
                            character.Class,
                            character.Race,
                            f"{character.Spec} {character.Class}"]

def create_image(image_prompt:str, character:Character, choice:str = "stable diffusion"):
    """MODELS"""
    
    if choice.strip() == "stable diffusion" or choice.strip() == "":
        create_image_stable_diffusion(image_prompt, character)
    elif choice.strip() == "dall-e":
        create_image_dall_e(image_prompt, character)

def create_image_stable_diffusion(image_prompt:str, character:Character):
    """creates an image using Stable Diffusion 2.1 """
    image_prompt = image_prompt.replace("\n","").strip()
    print("\rGenerating Stable Diffusion image. Please wait...", end="")
    websock = websocket.WebSocket()
    session_hash = str(uuid.uuid1()) # "wul4wr1g9t"
    websock.connect("wss://stabilityai-stable-diffusion.hf.space/queue/join")
    websock.send(json.dumps({"session_hash":session_hash,"fn_index":3}))
    images = []
    keywords = get_keywords(character) + [
                    #"3D render",
                    #"figure",
                    #"pose",
                    #"full figure",
                    #"full body and face",
                    "stylized",
                    "detailed",
                    "fantasy",
                    "character",
                    "colorful",
                    "high quality",
                    "stylized fantasy",
                    "low-detailed background landscape",
                    "low-detailed landscape",
                    "landscape",
                    "high-detailed foreground",

                ]
    prompt = "styles: " + ", ".join(keywords) + image_prompt
    spinner_chars = itertools.cycle(['-', '\\', '|', '/'])
    while True:
        sys.stdout.write(next(spinner_chars))
        sys.stdout.flush()
        message = websock.recv()
        if  "send_data" in message:
            websock.send(json.dumps({"fn_index":3,"data":[f"{prompt}","",9],"session_hash":session_hash}))
        if "process_completed" in message:
            res = json.loads(message)["output"]["data"][0]
            for image in res:
                images.append(image[23:])
            break
        sys.stdout.write("\b")

    for i,image in enumerate(images):
        with open(f"{os.path.join(IMAGE_PATH, character.Name.replace(' ','-'))}_{i}.jpg", "wb") as image_f:
            image_f.write(base64.decodebytes(bytes(image, "utf8")))
    websock.close()
    print(f"\rsaved to {os.path.join(IMAGE_PATH, character.Name.replace(' ','-'))}.jpg")

def create_image_dall_e(image_prompt:str, character:Character):
    """creates an image using dall-e"""
    print("\rGenerating DALL-E image. Please wait...", end="")
    image_prompt = image_prompt.replace("\n", "").strip()
    prompt_keywords = [
                        "simple",
                        #"stylized",
                        #"low-detailed",
                        #"portrait",
                        "detailed",
                        "fantasy",
                        "3D render",
                        #"pencil lines",
                        "clean",
                        "cartoony",
                        "colorful",
                        "figure",
                        ]
    pre_prompt = ", ".join(prompt_keywords) + \
                f", {character.Race}, "\
                f"{character.Class}, "\
                f"{character.Presenting_gender}"

    if len(image_prompt)+(len(pre_prompt)+1) > 1000:
        #print("orignal image prompt too long.")
        #print(f"old prompt: {image_prompt}")
        #print(f"old length: {len(image_prompt)}")
        image_prompt = image_prompt[0:999-(len(pre_prompt)+1)]
        #print(f"new prompt: {image_prompt}")
        #print(f"new length: {len(image_prompt)}")
        #input("enter to continue...")

    prompt = f"{pre_prompt}. {image_prompt}"
    sizes = {
        "large": "1024x1024",
        "medium": "512x512",
        "small": "256x256",
    }

    resp = openai.Image.create(
        prompt=prompt,
        n=1,
        size=sizes["large"]
    )
    img = resp["data"][0]["url"]
    download_image(img, character)

def create_backstory(character:Character):
    """
    creates a backstory for the character based on keywords
    """
    print("\rGenerating story. Please wait...", end="")
    keywords = get_keywords(character)
    #DEBUG(keywords)
    keywords_prompt = ", ".join(keywords)
    prompt = "the backstory has to be written for the perspectiv of the character. "\
            "Words such as \"DPS\", \"Tank\" and \"Healer\" refers to the characters role. "\
                "so if DPS, the character is a damage dealer, and so forth. "\
                "so dont say that the character is a DPS, instead say they are " \
                    "\"adept at fighting\" or if its a tank, "\
                    "then they're \"good at protection people\". "\
            "Avoid saying the name \"World of Warcraft\". "\
            "Please note that the world the character lives in is called \"Azeroth\". "\
            "Finish the backstory with the \"END BACKSTORY\". "\
            "create a short character backstory. "\
            "Start by saying \"i am [NAME] ... "\
            "the charaters short description is: "\
            f"I'm {character.Name}, "\
                f"a {character.Presenting_gender} {character.Race_description} {character.Class} "
    if get_current_image_model() == "stable diffusion":
        prompt = "be as visually desciptive as possible. "\
                    "Be as detailed as possible, describe, the look, armor, "\
                        "enviroment and everything else that could be used to descibe the "\
                            "visuals. And more than anything, describe the race as "\
                                "detailed as possible. "\
                                "I want you to literally describe it like a picture to a "\
                                    "blind person. "\
                    "Be descriptive about how the armor looks like. is it leather, mail, metal or cloth?. "\
                        "is it magical, or are prety gems sown in to the fabric? "\
                    "Describe the surrounding area around the character. "\
                    "be sure to describe how the character looks "\
                        "by describing the armor/clothes the character is wearing, and make the "\
                        "background reflect the culture of the character. " + prompt
    if get_current_image_model() == "dall-e":
        prompt = "The backstory cannot be longer than 255 "\
                "symbols (letters, spaces, etc...). " + prompt
    prompt = f"use these keywords for the backstory: {keywords_prompt}. " + prompt
    resp = openai.Completion.create(
        model=models["davinci-003"],
        prompt=prompt,
        max_tokens=1000,
        stop=["END"],
        n=1, # generates n backstories
        temperature=0.7,
    )

    with open("backstories.log", "a", encoding="utf8") as f:
        f.write(f"{character.Name}\n{resp.choices[0].text}\n{'#'*30}\n")
    return resp.choices[0].text
