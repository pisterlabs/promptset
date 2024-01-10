import logging
import time
from time import sleep

import pyrootutils
from langchain.llms import OpenAIChat

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)


class ChatGPT:
    task: str = "text2text-generation"

    def __init__(
        self, model_path: str = "gpt-3.5-turbo-0613", max_tokens: int = None
    ) -> None:
        # we'll just use langchain and build a wrapper around it such that it can be used like the HF pipelines
        self.chat = OpenAIChat()
        # NOTE: the model_path parameter is just there to avoid crashing during hydra instantiating with configs that pass model_path
        # Do not change this hardcoded value below!
        self.model_path = "gpt-3.5-turbo-0613"
        if max_tokens is not None:
            self.chat.model_kwargs["max_tokens"] = max_tokens

    def generate(self, inputs: list[str], rpm: int = 3500):
        responses = []
        for prompt in inputs:
            try:
                start = time.time()
                r = self.chat(prompt)
                # print(f"{prompt} {r}")
                responses.append(r)
                # sleep for enough time such that at least 20 seconds have passed since the last request
                diff = time.time() - start
                rate_limiting_time = 60 / rpm
                sleep_time = max(0, rate_limiting_time - diff)
                sleep(sleep_time)
            except Exception as e:
                # this may happen because we are out of credits or something else
                log.warning(str(e))
                responses.append(str(e))

        return responses

    def __call__(self, prompt: str, **kwargs):
        response = self.generate([prompt], **kwargs)
        return [{"generated_text": response[0]}]


if __name__ == "__main__":
    chat_gpt = ChatGPT(max_tokens=1)
    chat_gpt.generate(
        [
            "Please consider the following multiple-choice question and the four answer options A, B, C, and D.\nQuestion: An integer c is a common divisor of two integers x and y if and only if c is a divisor of x and c is a divisor of y. Which of the following sets of integers could possibly be the set of all common divisors of two integers?\nA: {-6,-2, -1, 1, 2, 6}\nB: {-6, -2, -1, 0, 1, 2, 6}\nC: {-6, -3, -2, -1, 1, 2, 3, 6}\nD: {-6, -3, -2, -1, 0, 1, 2, 3, 6}\n\nIf you were a philosophy expert, which answer would you choose? Answer: The answer is option"
        ]
    )
