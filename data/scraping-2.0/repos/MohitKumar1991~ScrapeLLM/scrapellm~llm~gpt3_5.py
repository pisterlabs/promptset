import openai
import time
from asyncer import asyncify
import scrapellm.settings as config

class GPT3_5:
    def __init__(
        self,
        *,
        model_params = {}
    ):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost: float = 0
        
        self.model_params = model_params
        openai.organization = openai.organization if openai.organization is not None else config.openai_org
        openai.api_key = openai.api_key if openai.api_key is not None else config.openai_key
        # default temperature to 0, deterministic
        if "temperature" not in model_params:
            model_params["temperature"] = 0
    
    async def call_raw_api_async(
            self, system_messages: list[str], user_messages: list[str], temperature=0
    ):
        return await asyncify(self.call_raw_api)(
                    user_messages=user_messages,
                    system_messages=system_messages,
                    temperature=temperature
            )

    def stream_raw_api(
            self, system_messages: list[str], user_messages: list[str], temperature=0
    ):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                        { "role": "system", "content": msg } for msg in system_messages
                     ]
                    + 
                    [
                        { "role": "user", "content": msg } for msg in user_messages
                    ],
            temperature=temperature,
            stream=True
        )
        for completion in response:
            choice = completion.choices[0]
            print(f"GOT CHUNK :: {choice.delta}")
            yield choice.delta

    def call_raw_api(
        self, system_messages: list[str], user_messages: list[str], temperature=0
    ):
        start_t = time.time()
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                        { "role": "system", "content": msg } for msg in system_messages
                     ]
                    + 
                    [
                        { "role": "user", "content": msg } for msg in user_messages
                    ],
            temperature=temperature,
        )
        elapsed = time.time() - start_t
        p_tokens = completion.usage.prompt_tokens
        c_tokens = completion.usage.completion_tokens
        
        choice = completion.choices[0]
        if choice.finish_reason != "stop":
            raise Exception(
                f"OpenAI did not stop: {choice.finish_reason} "
                f"(prompt_tokens={p_tokens}, "
                f"completion_tokens={c_tokens})"
            )
        print(choice.message.content, f"---Elapsed {elapsed} | p_tokens {p_tokens} ------- |  c_tokens {c_tokens}  -----\n\n") 
        return choice.message.content

    def get_embedding(self, text, model):
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], engine=model)["data"][0]["embedding"]