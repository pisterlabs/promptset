from typing import Literal
from aiogram.types import Message

import openai




class ChatGPT:

    def __init__(self, openai_token):
        self.api_key = openai_token
        
        openai.api_key = self.api_key
        
        self.max_tokens = 128
        self.temperature = 0
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0

        


    @staticmethod
    async def send_error(message: Message, kb):
        await message.answer(
            "⚠️ Произошла ошибка, попробуйте позже",
            reply_markup=kb,
        )


    @staticmethod
    def define_prompt_type(
            prompt_type: Literal['neg', 'pos'],
            lang = 'ru'
        ):
        
        prompt_types = {
            'ru': {
                "neg": "негативные",
                "pos": "позитивные",
            },
            'en': {
                "neg": "negative",
                "pos": "positive",
            },
        }
        
        return prompt_types[lang][prompt_type]


    @staticmethod
    def define_max_tokens(model_engine: str):
        model_tokens = {
            "gpt": 128,
            "gpt-3.5-turbo": 2800,
        }

        return model_tokens[model_engine]


    async def generate_answer(
            self,
            prompt: str,
            model_engine: str = "gpt",
            max_tokens = None,
        ):

        if not max_tokens:
            max_tokens = self.define_max_tokens(model_engine)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"{prompt}"},
            ],
            max_tokens=max_tokens
        )

        return completion
    

    async def get_answer(self, prompt: str, model_engine: str = "gpt", max_tokens = None) -> str:
        completion = await self.generate_answer(prompt, model_engine, max_tokens)
        answer: str = completion.choices[0].message.content

        answer = answer.replace("'", "`")
        answer = answer.replace('"', "`")
        
        return str(answer)
    

    async def get_extra_prompt(self, prompt: str, prompt_type: Literal['neg', 'pos'], amount: int = 10):
        
        prompt_type = self.define_prompt_type(prompt_type)
        prompt = f"На английском дай через запятую {amount} {prompt_type} промпты по запросу = {prompt}"
        extra_prompt: str = await self.get_answer(prompt=prompt, model_engine= "gpt-3.5-turbo")
        
        return extra_prompt
    