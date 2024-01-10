import os
from openai_function_call import OpenAISchema
from pydantic import Field
import openai
import time
from requests.exceptions import ProxyError
import requests
from PIL import Image
from io import BytesIO

openai.api_type = "azure"
openai.api_key = "0396650923ad40b880bf2a3ce3b80b9b"
openai.api_base = "https://myshell0.openai.azure.com"
openai.api_version = "2023-07-01-preview"

class chat_content(OpenAISchema):
    """
    Some descriptive text that will be used to create high-quality Stable Diffusion prompts.
    """
    content:str=Field(
        '',
        description="Some descriptive text that will be used to create high-quality Stable Diffusion prompts."
    )

class SDprompts(OpenAISchema):
    """
    Based on my text or description, create several high-quality Stable Diffusion prompts.
    """
    sd_prompt_1:str=Field(
        '',
        description="Stable diffusion prompt 1, slightly different from other prompts"
    )
    sd_prompt_2:str=Field(
        '',
        description="Stable diffusion prompt 2, slightly different from other prompts"
    )
    sd_prompt_3:str=Field(
        '',
        description="Stable diffusion prompt 3, slightly different from other prompts"
    )
    sd_prompt_4:str=Field(
        '',
        description="Stable diffusion prompt 4, slightly different from other prompts"
    )

stable_diffusion_prompt = """ """

def Call_SD(chat_content):

