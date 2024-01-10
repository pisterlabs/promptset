import openai
from PIL import Image
from typing import NoReturn, Literal, Union
from io import BytesIO
from openai import OpenAI, OpenAIError
from config.config import OPENAI_API
import asyncio
import base64
import json
import aiohttp
import aiofiles

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API)
