from model.model import Model
import openai
from tenacity import (retry,stop_after_attempt,wait_random_exponential)
from typing import List, Dict
import tiktoken, os

if os.name == 'nt':
    tiktoken_cache_dir = "C:/python/openai/rec_service/config/tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))
class GPT_turbo_model(Model):
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.prompt_template = cfg['prompt_template'] if 'prompt_template' in cfg else None
        self.MAX_CHARS = 8000

    @staticmethod
    def estimate_tokens(prompt: List[Dict]):
        '''
        Will estimate the number of tokens in the prompt
        :param prompt:
        :return:
        '''
        contents = [str(prompt_part['content']) for prompt_part in prompt]
        prompt_combined = " ".join(contents)
        enc = tiktoken.get_encoding("cl100k_base")
        encoded = enc.encode(prompt_combined)
        return len(encoded)
    @staticmethod
    def split_chunks(prompts, max_tokens: 5000):
        """Yield successive n-sized chunks from lst."""
        chunks = []
        all_ids = []
        chunk = []
        chunk_ids = []
        chunk_tokens = 0
        MAX_CHUNK_ITEMS = 19
        for i, prompt in enumerate(prompts):
            prompt_token_len = GPT_turbo_model.estimate_tokens(prompt)

            if (chunk_tokens + prompt_token_len) > max_tokens or len(chunk) > MAX_CHUNK_ITEMS:
                chunks.append(chunk)
                all_ids.append(chunk_ids)
                chunk_tokens = prompt_token_len
                chunk = [prompt]  # append to next chunk
                chunk_ids = [i]

            else:
                chunk.append(prompt)
                chunk_ids.append([i])
                chunk_tokens = chunk_tokens + len(prompt)
        if chunk:
            all_ids.append(chunk_ids)
            chunks.append(chunk)

        return chunks, all_ids

    def generate(self, prompts: List[List[Dict]],temp: int = 0.0, functions: List[Dict] = None):
        '''
        Will generate content
        :param text:
        :return:
        '''
        if os.name == 'nt':
            os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/openai.crt"
        if self.prompt_template is not None:
            prompts =[ f"{self.prompt_template} {text}"
                       for text in prompts
                       ]

        all_answers = []

        chunks, all_ids = GPT_turbo_model.split_chunks(prompts,
                                                       max_tokens=self.MAX_CHARS)

        all_chunk_ids = []
        for chunk, chunk_ids in zip(chunks, all_ids):
            all_chunk_ids = all_chunk_ids + chunk_ids
            answers = self.call_n_parse(
                prompts=chunk,
                functions=functions,
                config=self.cfg,
                temp=temp
            )
            if all_answers is None:
                all_answers = answers
            else:
                all_answers = all_answers + answers
        assert len(prompts) == len(all_chunk_ids)
        return all_answers

    def call_n_parse(self, prompts, functions, config, temp: int = 0.0):
        '''
        Will call GPT turbo model and parse the results
        :param prompts:
        :param functions:
        :param config:
        :param temp:
        :return:
        '''
        answers = []
        for prompt_messages in prompts:
            response = self.call_chatgpt(prompt_messages, functions, config, temp)

            for choice in response.choices:
                text = choice.message.content
                answers.append(text)

        assert len(prompts) == len(answers)
        return answers


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def call_chatgpt(self,prompt_messages,functions, config: Dict = None, temp: int = 0.0):
        '''
        Will call GPT turbo model
        prompts is a List of messages
        :param prompt_messages:
        :param functions:
        :param config:
        :param temp:
        :return:
        '''
        if config is None:
            config = self.cfg
        openai.api_key = os.environ['OPENAI_API_KEY']
        model_name = config['model_name'] if 'model_name' in config else "gpt-4"
        max_tokens = config['max_tokens'] if 'max_tokens' in config else 4000
        temperature = config['temperature'] if 'temperature' in config else 0.0
        print(prompt_messages)
        if functions is not None:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=prompt_messages,
                functions=functions,
                temperature=temp if temp !=0.0 else temperature,
                max_tokens=max_tokens
            )
        else:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=prompt_messages,
                temperature=temp if temp !=0.0 else temperature,
                max_tokens=max_tokens
            )
        return response
