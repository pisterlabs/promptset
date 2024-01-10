import logging
import time
from time import sleep

import pyrootutils
from langchain.llms import VertexAI
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)


class PaLM:
    task: str = "text2text-generation"

    def __init__(
        self,
        model_path: str = "text-bison",
        max_tokens: int = None,
        temperature: float = 0.7,
    ) -> None:
        # NOTE: the model_path parameter is just there to avoid crashing during hydra instantiating with configs that pass model_path
        # Do not change this hardcoded value below!
        self.model_path = "text-bison"
        # we'll just use langchain and build a wrapper around it such that it can be used like the HF pipelines
        model_kwargs = {"model_name": self.model_path, "temperature": temperature}
        if max_tokens is not None:
            model_kwargs["max_output_tokens"] = max_tokens
        self.chat = VertexAI(**model_kwargs)

    def generate(self, inputs: list[str], rpm: int = 20, max_tries: int = 10):
        responses = []
        rate_limiting_time = 60 / rpm
        for prompt in tqdm(inputs):
            try:
                assert max_tries > 0
                for i in range(max_tries):
                    start = time.time()
                    # import pdb

                    # pdb.set_trace()
                    r = self.chat(prompt)
                    # print(f"{prompt} {r}")
                    # sleep for enough time such that at least 20 seconds have passed since the last request
                    diff = time.time() - start
                    sleep_time = max(0, rate_limiting_time - diff)
                    sleep(sleep_time)

                    if r != "":
                        break

                if i == max_tries - 1:
                    print(f"Exhausted maximum tries: {i+1}")

                responses.append(r)
            except Exception as e:
                # this may happen because we are out of credits or something else
                log.warning(str(e))
                responses.append(str(e))

        return responses

    def __call__(self, prompt: str, **kwargs):
        response = self.generate([prompt], **kwargs)
        return [{"generated_text": response[0]}]


if __name__ == "__main__":
    palm = PaLM(max_tokens=1)
    response = palm.generate(
        [
            "Please consider the following multiple-choice question and the four answer options A, B, C, and D.\nQuestion: An integer c is a common divisor of two integers x and y if and only if c is a divisor of x and c is a divisor of y. Which of the following sets of integers could possibly be the set of all common divisors of two integers?\nA: {-6,-2, -1, 1, 2, 6}\nB: {-6, -2, -1, 0, 1, 2, 6}\nC: {-6, -3, -2, -1, 1, 2, 3, 6}\nD: {-6, -3, -2, -1, 0, 1, 2, 3, 6}\n\nIf you were a philosophy expert, which answer would you choose? Answer: The answer is option"
        ]
    )
    print(response)
