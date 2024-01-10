import requests
import json
import pdb
import os
import re

from typing import Dict, List, Optional, Union

import autogen
from autogen.agentchat import Agent, ConversableAgent

from autogen.agentchat.contrib.img_utils import _to_pil
import urllib.request
from openai import OpenAI
import os
import PIL

config_list_dalle = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["dalle"],
    },
)

from diskcache import Cache

def dalle_call(client: OpenAI, model: str, prompt: str, size: str, quality: str, n: int) -> str:
    """
    Generate an image using OpenAI's DALL-E model and cache the result.

    This function takes a prompt and other parameters to generate an image using OpenAI's DALL-E model.
    It checks if the result is already cached; if so, it returns the cached image data. Otherwise,
    it calls the DALL-E API to generate the image, stores the result in the cache, and then returns it.

    Args:
        client (OpenAI): The OpenAI client instance for making API calls.
        model (str): The specific DALL-E model to use for image generation.
        prompt (str): The text prompt based on which the image is generated.
        size (str): The size specification of the image. TODO: This should allow specifying landscape, square, or portrait modes.
        quality (str): The quality setting for the image generation.
        n (int): The number of images to generate.

    Returns:
    str: The image data as a string, either retrieved from the cache or newly generated.

    Note:
    - The cache is stored in a directory named '.cache/'.
    - The function uses a tuple of (model, prompt, size, quality, n) as the key for caching.
    - The image data is obtained by making a secondary request to the URL provided by the DALL-E API response.
    """
    # Function implementation...
    cache = Cache('.cache/')  # Create a cache directory
    key = (model, prompt, size, quality, n)
    if key in cache:
        return cache[key]

    # If not in cache, compute and store the result
    response = client.images.generate(
          model=model,
          prompt=prompt,
          size=size,
          quality=quality,
          n=n,
        )
    image_url = response.data[0].url
    # print("Image URL:", image_url)
    # img_data = get_image_data(image_url)
    cache[key] = image_url
    
    return image_url

def extract_img(agent: Agent) -> PIL.Image:
    """
    Extracts an image from the last message of an agent and converts it to a PIL image.

    This function searches the last message sent by the given agent for an image tag,
    extracts the image data, and then converts this data into a PIL (Python Imaging Library) image object.

    Parameters:
        agent (Agent): An instance of an agent from which the last message will be retrieved.

    Returns:
        PIL.Image: A PIL image object created from the extracted image data.

    Note:
    - The function assumes that the last message contains an <img> tag with image data.
    - The image data is extracted using a regular expression that searches for <img> tags.
    - It's important that the agent's last message contains properly formatted image data for successful extraction.
    - The `_to_pil` function is used to convert the extracted image data into a PIL image.
    - If no <img> tag is found, or if the image data is not correctly formatted, the function may raise an error.
    """
    # Function implementation...
    img_data = re.findall("<img (.*)>", agent.last_message()["content"])[0]
    pil_img = _to_pil(img_data)
    return pil_img

class DALLEAgent(ConversableAgent):
    def __init__(self, name, llm_config: dict, **kwargs):
        super().__init__(name, llm_config=llm_config, **kwargs)
        
        try:
            config_list = llm_config["config_list"]
            api_key = config_list[0]["api_key"]
        except Exception as e:
            print("Unable to fetch API Key, because", e)
            api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.register_reply([Agent, None], DALLEAgent.generate_dalle_reply)
        
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # override and always "silent" the send out message; 
        # otherwise, the print log would be super long!
        super().send(message, recipient, request_reply, silent=True)
        
    def generate_dalle_reply(self, messages: Optional[List[Dict]], sender: "Agent", config):
        """Generate a reply using OpenAI DALLE call."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        prompt = messages[-1]["content"]

        # Manipulate response for images generation.
        image_descriptions = json.loads(prompt)
        urls = []
        for ele in image_descriptions:
            slide_title = ele.get("slide_title")
            image_description = ele.get("image_description")

            prompt = f"{self.system_message}\nSlide: {slide_title}\nDescription: {image_description}"
            # TODO: integrate with autogen.oai. For instance, with caching for the API call
            img_url = dalle_call(
                client=self.client,
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024", # TODO: the size should be flexible, deciding landscape, square, or portrait mode.
                quality="standard",
                n=1,
            )
            slide_title = slide_title.lower().replace(" ", "_").replace(".", "").replace(",", "").replace("-", "_")
            print(slide_title)
            filename = './images/'+slide_title+".jpg"
            urllib.request.urlretrieve(img_url, filename)
            urls.append(filename)
        final_images = []
        for ele, url in zip(image_descriptions, urls):
            final_images.append({"slide_title": ele.get("slide_title"), "url": url})
        print("final_images")
        print(final_images)
        with open('image.json', 'w') as f:
            json.dump(final_images, f)
        return True, "TERMINATE"
    
# dalle = DALLEAgent(name="Dalle", llm_config={"config_list": config_list_dalle})

# user_proxy = autogen.UserProxyAgent(
#     name="User_proxy",
#     system_message="A human admin.",
#     human_input_mode="NEVER", 
#     max_consecutive_auto_reply=0
# )

# Ask the question with an image
# user_proxy.initiate_chat(dalle,
#                          message="""Use a medical-themed infographic with icons such as a clipboard for "Inefficient data management," a clock for "High administrative workload," a magnifying glass for "Difficulty in early detection," a lock for "Privacy and security concerns," and a cloud with a lightning bolt for "Limited real-time insights." """, 
#                          )