"""Large language model chat with GPT API"""
import time
import openai
from typing import Dict, Any, Tuple, Union, List
from PIL import Image
import numpy as np
from lgmcts.utils.user import print_type_indicator
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed


class LLM:
    """Abstract class for large language model"""

    def __init__(self):
        self._is_api = False

    @property
    def is_api(self):
        return self._is_api

    @is_api.setter
    def is_api(self, is_api):
        self._is_api = is_api


class ChatGPTAPI(LLM):
    """Chat with GPT API"""

    def __init__(self, model: str = "gpt-4", api_key: str = "", db: str = None):
        super().__init__()
        self.is_api = True
        openai.api_key = api_key
        self.gpt_model = model
        self.system_prompt = {"role": "system", "content": db}

    def chat(
        self,
        str_msg: Union[str, List[Any]],
        img_msg: Union[List[Image.Image], List[np.ndarray], None] = None,
        **kwargs
    ) -> Tuple[str, bool]:
        # Print typing indicator
        print_type_indicator("LLM")
        if isinstance(str_msg, list):
            return self.talk_prompt_list(str_msg), True
        elif isinstance(str_msg, str):
            return self.talk_prompt_string(str_msg), True

    def _threaded_talk_prompt(self, prompt: Dict[str, Any], max_retries: int = 4) -> Tuple[str, Any]:
        # print("Threaded execution of prompt: {}".format(prompt))
        retries = 0
        conversation = [self.system_prompt]
        while retries <= max_retries:
            try:
                # assert len(prompt) == 1
                # for key, value in prompt.items():
                conversation.append({"role": "system", "content": prompt})
                reply = openai.ChatCompletion.create(
                    model=self.gpt_model,
                    messages=conversation,
                    timeout=10  # Timeout in seconds for the API call
                )
                reply_content = reply["choices"][0]["message"]["content"]
                conversation.append({"role": "assistant", "content": reply_content})
                return reply_content, None
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    time.sleep(10)  # Wait for 10 seconds before retrying
                else:
                    return None, str(e)

    def talk_prompt_list(self, prompt_list: List[Dict[str, Any]], batch_size: int = 4) -> List[str]:
        """prompt_list is a list of dict, each dict has one key and one value"""
        results = [None] * len(prompt_list)
        errors = []
        batch_size = min(100, len(prompt_list))
        for i in range(0, len(prompt_list), batch_size):
            # Create thread pool
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                print("Batch Execution of prompts {} to {}".format(i, min(i+batch_size-1, len(prompt_list))))
                future_to_prompt = {executor.submit(self._threaded_talk_prompt, prompt, max_retries=4)                                    : index for index, prompt in enumerate(prompt_list[i:i+batch_size])}

                for future in as_completed(future_to_prompt):
                    index = future_to_prompt[future]
                    try:
                        reply_content, error = future.result()
                        if reply_content is not None:
                            results[index] = reply_content
                        else:
                            errors.append(f"Error for prompt {prompt_list[index]}: {error}")
                    except TimeoutError:
                        errors.append(f"Timeout error for prompt {prompt_list[index]}")
            for error in errors:
                print(error)

        return results
