from .llm_interface import LLMInterface
from .llm_prompt_interface import LLMPromptInterface
from typing import List, Dict
import openai
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpenaiCalls")


class GPT35(LLMInterface):
    def __init__(self, api_key: str) -> None:
        self.history = []
        self.model = "gpt-3.5-turbo-0301"
        self.request_attemps = 3
        openai.api_key = api_key

    def __accept_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})

    def __get_next_responses(self, choices: int = 1) -> List[str]:
        completion = openai.ChatCompletion.create(
            model=self.model, 
            messages=self.history,
            n=choices
        )

        best_response = completion['choices'][0]['message']['content']
        self.history.append({"role": "assistant", "content": best_response})

        return [choice['message']['content'] for choice in completion["choices"]]

    def init_history(self, llm_prompt: LLMPromptInterface) -> None:
        prompt = llm_prompt.get_system_message()
        message_history = llm_prompt.get_msg_history()

        self.history.append( {"role": "system", "content": prompt} )
        self.history.extend(message_history)

    def send_message_for_response(self, message: str, choices: int = 1) -> List[str]:
        self.__accept_message(message)
        return self.__get_next_responses(choices=choices)

    def send_message_wout_history_change(self, message: str, choices: int = 1) -> List[str]:
        attempts = self.request_attemps
        while attempts > 0:
            try: 
                completion = openai.ChatCompletion.create(
                    model=self.model, 
                    messages=self.history + [{"role": "user", "content": message}],
                    n=choices
                )
                return [choice['message']['content'] for choice in completion["choices"]]
            except Exception as e:
                attempts -= 1
                logger.info(f"OpenAI API call failed with {e}. {attempts} attempts left.")
                if attempts == 0: 
                    raise e
                else: 
                    continue