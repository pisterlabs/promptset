from gimpfu import *
import os
import json
import urllib2
from base64 import b64decode
import mimetypes
import tempfile

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "gimp-dall-e")
CONFIG_FILE_NAME = "openai_key.json"
API_PATH = "https://api.openai.com/v1/images/generations"
MODEL_MAP = {
    0: "dall-e-2",
    1: "dall-e-3",
}

IMAGES_POSSIBLE_SIZE = {
        "256x256": "256x256",
        "512x512": "512x512",
        "1024x1024": "1024x1024",
        }

STYLES = {
        "vivid": "vivid",
        "natural": "natural",
        }

QUALITIES = {
        "standard": "standard",
        "hd": "hd",
        }

# validate if config dir does exists and, if not, create it
def create_config_dir():
    if not os.path.exists(CONFIG_DIR): os.makedirs(CONFIG_DIR)

# saves the openai api key in a json file
def save_openai_api_key(api_key):
    """
    Saves the OpenAI API key in a JSON file.
    Args:
        api_key: the OpenAI API key
    Returns:
        None
    """

    create_config_dir()
    j = {"api_key": api_key}
    with open(os.path.join(CONFIG_DIR, CONFIG_FILE_NAME), "w") as f: json.dump(j, f)

def mask_openai_api_key(api_key):
    """
    Masks the OpenAI API key.
    Args:
        api_key: the OpenAI API key
    Returns:
        The masked OpenAI API key
    """

    if get_openai_api_key() == "": return ""
    return api_key[:2] + "*" * (len(api_key) - 4) + api_key[-2:]

def get_openai_api_key():
    """
    Returns the OpenAI API key.
    Args:
        None
    Returns:
        The OpenAI API key
    """

    if os.path.exists(os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)):
        with open(os.path.join(CONFIG_DIR, CONFIG_FILE_NAME), "r") as f:
            return json.load(f).get("api_key", "")
    return ""

def create_image(model, size, style, quality, api_key, prompt, n):
    """
    Creates the image.
    Args:
        model: the model to use
        size: the size of the image
        style: the style to use
        api_key: the OpenAI API key
        prompt: the prompt to use
        n: the number of completions
    Returns:
        None
    """

    url = API_PATH
    data = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "style": style,
        "quality": quality,
        'response_format': 'b64_json'
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(api_key)
    }
    request = urllib2.Request(url, json.dumps(data), headers)
    pdb.gimp_progress_set_text("Creating image...")
    try:
        response = urllib2.urlopen(request)
        response_data = json.loads(response.read())
        if response_data.get("data"):
            out_prefix = tempfile.mktemp()
            for i, image in enumerate(response_data["data"]):
                image_data = b64decode(image["b64_json"])
                out_file_name = out_prefix + "_" + str(i) + ".png"
                with open(out_file_name, "wb") as f: f.write(image_data)
                pdb.gimp_progress_set_text("Loading image...")
                created = pdb.gimp_file_load(out_file_name, out_file_name)
                pdb.gimp_display_new(created)
                pdb.gimp_displays_flush()
                # os.remove(out_file_name)
                return
        else:
            gimp.message("Error: nothing returned from OpenAI.")
            return None
    except urllib2.HTTPError as e:
        # saves the error message to a temporary file

        out_file_name = "/tmp/error.txt"
        with open(out_file_name, "w") as f: f.write(e.read())
        gimp.message("Error: {}".format(e))
        return None

def dall_e(image, drawable, model, size, style, quality, api_key, prompt):
    model = MODEL_MAP[model]
    print("DALL-E plugin started...")

    if quality == "hd":
        if model == "dall-e-2":
            print("dall-e-2 does not support hd quality.")
            gimp.message("dall-e-2 does not support hd quality.")
            return

        if size != "1024x1024":
            print("hd quality only works with 1024x1024 size.")
            gimp.message("hd quality only works with 1024x1024 size.")
            return
    
    if (model == "dall-e-2") and (quality == "hd"):
        print("dall-e-2 does not support hd quality.")
        gimp.message("dall-e-2 does not support hd quality.")
        return

    if api_key[:3] == "sk*":
        print("Using saved OpenAI API key.")
        api_key = get_openai_api_key()
    elif api_key == "":
        print("No OpenAI API key set.")
        gimp.message("You need to set your OpenAI API key first.")
        return
    elif api_key[:2] != "sk":
        print("Invalid OpenAI API key.")
        gimp.message("Invalid OpenAI API key.")
        return
    else:
        print("Saving OpenAI API key.")
        save_openai_api_key(api_key)

    n = 1
    create_image(model, size, style, quality, api_key, prompt, n)

register(
    "python_fu_dall-e_create",
    "DALL-E plugin",
    "Edit and create images with the power of DALL-E",
    "Paulo Vinicius", "Paulo Vinicius", "2023",
    "Create",
    "",
    [        
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input layer", None),
        (PF_OPTION, "model", "Model", 0, tuple(MODEL_MAP.values())),
        (PF_RADIO, "size", "Image size", "512x512", tuple(IMAGES_POSSIBLE_SIZE.items())),
        (PF_RADIO, "style", "Style", "vivid", tuple(STYLES.items())),
        (PF_RADIO, "quality", "Quality", "standard", tuple(QUALITIES.items())),
        (PF_STRING, "api_key", "OpenAI Key", mask_openai_api_key(get_openai_api_key())),
        (PF_TEXT, "prompt", "Prompt", " "),
    ],
    [],
    dall_e, 
    menu="<Image>/DALL-E"
)

main()
