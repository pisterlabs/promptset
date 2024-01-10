from datetime import datetime
from itertools import product
import logging
import time
from typing import List
from disnake.ext import commands
import openai
import copy

from retry import retry


class ChatCompletionCog(commands.Cog):
    async def __init__(self,
                       bot: commands.Bot):
        self.bot = bot        

    def set_message_context(self, sys_prompt: str, usr_msg: List[str], ast_msg: List[str]):
        messages = [{"role": "system", "content": sys_prompt}]

        for i in range(max(len(usr_msg), len(ast_msg))):
            if i < len(usr_msg):
                messages.append({"role": "user", "content": usr_msg[i]})

            if i < len(ast_msg):
                messages.append({"role": "assistant", "content": ast_msg[i]})

        self.messagecontext = messages

    async def get_response(self, message: str, placeholder_strings: dict[str, str] = []) -> str:                  
        messages = copy.deepcopy(self.messagecontext)

        if placeholder_strings is not None and len(placeholder_strings) > 0:
            for msg, (placeholder, replacement) in product(messages, placeholder_strings.items()):
                msg['content'] = msg['content'].replace(placeholder, replacement)

        response = ""        
        messages.append({"role": "user", "content": message})

        response = self.__call_openai(messages)

        logging.info(f"Input: {message}\n\tResponse: {response}\n")        
        
        return response    
    
    @retry(tries=3, delay=5, backoff=5, logger=logging.getLogger(__name__))
    def __call_openai(self, messages) -> str:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages
        )

        return completion.choices[0].message.content