import pandas as pd
from prompt_lib.backends import openai_api

from src.utils import Prompt


class GSMFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 900) -> None:
        super().__init__(
            question_prefix="# Q: ",
            answer_prefix="# A: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###n\n",
            engine = engine,
            temperature = temperature
        )
        self.max_tokens = max_tokens
        self.instruction = "# There is an error in the above reasoning chain due to lack of understanding of the concept. What is the error? To check for the error, go through each step in the reasoning chain and check for these three factors: 1. Is the step relevant to solving the problem? 2. Is the step logically correct? 3. Is the step mathematically correct?"
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()
    
    def __call__(self, question: str, solution: str):
        generation_query = self.make_query(question=question, solution=solution)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="### END ###",
            temperature=self.temperature,
        )
        
        entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
        # print(f"Feedback Output: {entire_output}")
        if "### END" in entire_output:
            entire_output = entire_output.split("### END")[0]
        feedback = entire_output
        return feedback

    def make_query(self, question: str, solution: str):
        solution = f"""{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}{solution}{self.intra_example_sep}{self.instruction}{self.inter_example_sep}Feedback:"""
        return f"{self.prompt}{solution}"
    

def test():
    task_fb = GSMFeedback(
        prompt_examples="data/prompt/gsm_nl/feedback.txt",
        engine="text-davinci-003",
        temperature=0.7,
    )

    question = "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
    wrong_soln = """Let's break down this problem: 1. How many toys did Shawn have before Christmas? 2. How many toys did Shawn get on Christmas? 3. How many toys does Shawn have now?
1. Shawn had 5 toys before Christmas.
2. Shawn got 2 toys each from his mom and dad. So, he got 2 toys on Christmas.
3. Shawn has 5 + 2 = 7 toys.
The answer is 7."""
    feedback = task_fb(question, wrong_soln)
    print(feedback)
    

if __name__ == '__main__':
    test()