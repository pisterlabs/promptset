import sys
from typing import Dict, List
from src.utils import Prompt

from prompt_lib.backends import openai_api

class GSMIterate(Prompt):
  def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 500) -> None:
    super().__init__(
      question_prefix="# Q: ",
      answer_prefix="# A: Let's break down this problem: ",
      intra_example_sep="\n\n",
      inter_example_sep="\n\n",
    )
    self.engine = engine
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.instruction = "# Given the above feedback on the answer and the reasoning chain, regenerate the correct reasoning chain and answer. Make sure to incorporate all the feedback."
    self.setup_prompt_from_examples_file(prompt_examples)
  
  def setup_prompt_from_examples_file(self, examples_path: str) -> str:
    with open(examples_path, "r") as f:
      self.prompt = f.read()
  
  def __call__(self, question: str, solution: str, feedback: str) -> str:
    generation_query = self.make_query(question=question, solution=solution, feedback=feedback)
    output = openai_api.OpenaiAPIWrapper.call(
      prompt=generation_query,
      engine=self.engine,
      max_tokens=self.max_tokens,
      stop_token="### END ###",
      temperature=self.temperature
    )

    entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
    # print(f"Iterate Output: {entire_output}")
    if "### END ###" in entire_output:
      entire_output = entire_output.split("### END ###")[0].strip()
    solution = entire_output.split(self.answer_prefix)[1].strip()
    return solution
  
  def make_query(self, question: str, solution: str, feedback: str) -> str:
    solution = f"""{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}{solution}{self.intra_example_sep}Feedback: {feedback}{self.intra_example_sep}{self.instruction}"""
    query = f"{self.prompt}{self.inter_example_sep}{solution}"
    return query
  
def test():
  task_iterate = GSMIterate(
    engine="text-davinci-003",
    prompt_examples="data/prompt/gsm_nl/iterate.txt",
    temperature=0.7
  )

  question = "Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup."
  wrong_soln = """1. How much half a dozen plates cost? 2. What is the cost of buying twenty dozen cups? 3. What is the cost of one cup?
1. Half a dozen plates cost $6000. So each plate costs $6000 / 6 = $1000.
2. Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000. So twenty dozen cups cost $6000 - $1200 = $4800.
3. Total cups = 20 * 12 = 240. So each cup costs $4800 / 240 = $20.
The answer is $20."""
  feedback = """Let us go through step-by-step and check the three error categories.

# Step 1: How much does half a dozen plates cost? Half a dozen plates cost $6000. So each plate costs $6000 / 6 = $1000.
## 1. This step is relevant because we need to calculate the cost of half dozen plates to calculate cost of 20 dozens cups.
## 2. This step is logically incorrect. Each plate costs $6000, so we need to multiply the cost of each plate with 6 to get the cost of half a dozen plates. So half a dozen plates costs $6000 * 6 = $36000.
## 3. This step is mathematically correct.

# Step 2: How much does twenty dozen cups cost? Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000. So twenty dozen cups cost $6000 - $1200 = $4800.
## 1. This step is relevant because we need to calculate the cost of twenty dozen cups to calculate the cost of one cup.
## 2. This step is logically incorrect because of the incorrect solution of step 1. The total cost of half a dozen plates sold at $6000 each is $36000, not $6000. So twenty dozen cups cost $36000 - $1200 = $34800.
## 3. This step is mathematically correct.

# Step 3: How much does one cup cost? Total cups = 20 * 12 = 240. So each cup costs $4800/ 240 = $20.
## 1. This step is relevant because we need to calculate the cost of one cup to answer the question.
## 2. This step is logically incorrect because of the incorrect solution of step 2. Cost of twenty dozen cups is $34800, not $4800. So each cup costs $34800 / 240 = $145.
## 3. This step is mathematically correct."""
  print(task_iterate(question, wrong_soln, feedback))

if __name__ == "__main__":
  test()