import openai

from bots.bot import Bot
from bots.type import BotType
from config import CONFIG

class GPTBot(Bot):
    def __init__(self):
        super().__init__(BotType.GPT)
        self.model = "gpt-3.5-turbo"
        self.messages = []
        
        openai.api_key = CONFIG.openai_api_key

    def _call_api(self, prompt: str):
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature = 0
        )
    
        return response['choices'][0]['message']['content']