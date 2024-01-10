# MISCELLANEOUS HELPERS

import numpy as np
import ipywidgets as widgets
from ipywidgets import interact_manual
import openai
import validators
from PIL import Image
from io import BytesIO
import requests
import base64

from .msb_enums import idMode


# ID creation --------------------------------------
# Function to create a unique id for a dataset, concept or sketch, with format [d/c/s]_[index]
def create_id(
    sb,
    mode,
):
    if mode == idMode.Dataset:
        new_index = len(sb.datasets)
        return f"d_{new_index}"
    elif mode == idMode.Concept:
        new_index = len(sb.concepts)
        return f"c_{new_index}"
    elif mode == idMode.Sketch:
        new_index = len(sb.sketches)
        return f"s_{new_index}"


# DataFrame styling helper functions --------------------------------------
# Function to add red and green colors to Pandas dataframe background (for binary scores)
def color_t_f(df):
    return [
        "background-color: lightgreen" if x else "background-color: salmon" for x in df
    ]


# Function to add yellow color of varying opacity to Pandas dataframe background (for continuous 0 to 1 scores)
def color_magnitude(df):
    return [
        f"background-color: rgba(255,215,0,{x})" if x > 0 else "background-color: white"
        for x in df
    ]


# Function to add red/blue color of varying opacity to Pandas dataframe background (for continuous -1 to 1 scores)
def color_pos_neg(df):
    # red for negative; rgba(240,100,100)
    # blue for positive; rgba(100,150,240)
    return [
        f"background-color: rgba(240,100,100,{abs(x)})"
        if x < 0
        else f"background-color: rgba(100,150,240,{abs(x)})"
        for x in df
    ]


def rescale_img(
    img,
    max_width=200,
):
    w, h = img.size
    if w < max_width and h < max_width:
        return img

    if w > h:
        new_w = max_width
        new_h = np.floor(h * (max_width / w))
    else:
        new_h = max_width
        new_w = np.floor(w * (max_width / h))

    return img.resize((int(new_w), int(new_h)))


# Helper to fetch an HTML img tag with a base64-encoded image given an image URL (for display in DataFrame in NB)
# - path: String URL of an image
# - max_width: Maximum width of image in pixels
def path_to_image_html(path: str, max_width: int = 250):
    if not validators.url(path):
        return f'<img src="{path}" style="max-width: {max_width}px;">'
    else:
        img = Image.open(BytesIO(requests.get(path).content))
        img = rescale_img(img, max_width)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
        return f'<img src="data:image/png;base64,{img_str}" style="max-width: {max_width}px;">'


# Open AI Credentials --------------------------------------
# Function to set OpenAI credentials
def set_openai_credentials_internal(
    organization,
    api_key,
):
    openai.organization = organization
    openai.api_key = api_key
    print("Saved!")


# Function to show a widget for getting OpenAI credentials
def set_openai_credentials():
    organization = widgets.Text(
        value=None,
        placeholder="Enter OpenAI organization",
        description="OpenAI organization: ",
        disabled=False,
        style=dict(description_width="initial"),
    )

    api_key = widgets.Text(
        value=None,
        placeholder="Enter OpenAI API key",
        description="OpenAI API key: ",
        disabled=False,
        style=dict(description_width="initial"),
    )

    widgets.interact_manual.opts["manual_name"] = "Set credentials"

    interact_manual(
        set_openai_credentials_internal,
        organization=organization,
        api_key=api_key,
    )

    widgets.interact_manual.opts["manual_name"] = "See results"
