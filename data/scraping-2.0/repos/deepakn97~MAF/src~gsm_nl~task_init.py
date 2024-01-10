import pandas as pd
from src.utils import Prompt

from prompt_lib.backends import openai_api


class GSMInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str, temperature: float, max_tokens: str = 500) -> None:
        super().__init__(
            question_prefix="# Q: ",
            answer_prefix="# A: Let's break down this problem:",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples) -> str:
        with open(prompt_examples, "r") as f:
            self.prompt = f.read()
    
    def make_query(self, question: str) -> str:
        question = question.strip()
        query = f"{self.prompt}{self.inter_example_sep}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        return query

    def __call__(self, question: str) -> str:
        generation_query = self.make_query(question)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="# Q:",
            temperature=self.temperature,
        )

        solution = openai_api.OpenaiAPIWrapper.get_first_response(output)
        # print(f"Init Solution: {solution}")

        return solution.strip()


def test():
    task_init = GSMInit(
        prompt_examples="data/prompt/gsm_nl/init.txt",
        engine="gpt-3.5-turbo",
        temperature=0.0,
    )

    question = "The educational shop is selling notebooks for $1.50 each and a ballpen at $0.5 each.  William bought five notebooks and a ballpen. How much did he spend in all?"
    print(task_init(question))
    

if __name__ == "__main__":
    test()