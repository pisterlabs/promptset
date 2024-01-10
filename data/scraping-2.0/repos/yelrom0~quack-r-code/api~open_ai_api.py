"""
open_ai_api.py
The backend openAI/chatGPT functions of this application.
Most of this code is repurposed from my previously written chat bot
"""

# System Imports

# Package Imports
import openai
from dotenv import dotenv_values


class AIApi:
    def __init__(self):
        # Load environment variables and prompt file
        self.env = dotenv_values(".env")
        chat_gpt_prompt_file = open("./prompt.txt", "r")

        # Constants
        self.SYSTEM_PROMPT = chat_gpt_prompt_file.read()
        self.FILE = open("debug_log.txt", "w")

        # init openai interface
        openai.api_key = self.env["OPENAI_API_KEY"]
        # self.chat_completion = openai.ChatCompletion(engine="gpt-3.5-turbo")

    async def get_stable_diffusion_prompt(self, text: str) -> str:
        self.messages = [
            {"role": "system", "content": f"{self.SYSTEM_PROMPT}"}
        ]
        # get response from openai interface
        self.messages.append({"role": "user", "content": f"{text}"})

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0,
        )
        response = response.choices[0].message.content
        return response
