import sys
import time
from typing import Dict, List
from src.utils import Prompt

from prompt_lib.backends import openai_api

class GSMIterate(Prompt):
  def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 300) -> None:
    super().__init__(
      question_prefix="",
      answer_prefix="",
      intra_example_sep="\n\n",
      inter_example_sep="\n\n",
    )
    self.engine = engine
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.instruction = "# Given the feedback and the original code, let's rewrite the code to incorporate all of the feedback."
    self.setup_prompt_from_examples_file(prompt_examples)
  
  def setup_prompt_from_examples_file(self, examples_path: str) -> str:
    with open(examples_path, "r") as f:
      self.prompt = f.read()
  
  def __call__(self, solution: str, feedback: str) -> str:
    generation_query = self.make_query(solution=solution, feedback=feedback)
    # print(generation_query)
    success = False
    while not success:
        try:
            output = openai_api.OpenaiAPIWrapper.call(
                prompt=generation_query,
                engine=self.engine,
                max_tokens=self.max_tokens,
                stop_token="### END ###",
                temperature=self.temperature,
            )
            success = True
        except Exception as e:
            success = False
            print(e)
            time.sleep(60)

    entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
    # print(f"Iterate Output: {entire_output}")
    if "### END ###" in entire_output:
      entire_output = entire_output.split("### END ###")[0].strip()
    solution = entire_output.split("def solution():")[1]
    solution = "def solution():" + solution.rstrip()
    return solution
  
  def make_query(self, solution: str, feedback: str) -> str:
    solution = f"""{solution}{self.intra_example_sep}Feedback:\n{feedback}{self.intra_example_sep}{self.instruction}"""
    query = f"{self.prompt}{self.intra_example_sep}{solution}"
    return query
  
def test():
  task_iterate = GSMIterate(
    engine="text-davinci-002",
    prompt_examples="data/prompt/gsm_iter/iterate.txt",
    temperature=0.7
  )

  wrong_soln = """def solution():
  \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = plate_cost
    result = cup_cost
    return result"""
  feedback = """# Let us go through the error and check step-by-step
    plates = 6
    plate_cost = 6000
# looks good

# Let's check the other parts
    cups = 12 * 20
    cup_cost = plate_cost
# wrong! The cost of a cup is not the same as the cost of a plate. The cost of a cup is $1200 less than the total cost of half a dozen plates sold at $6000 each. So we need to calculate the cost of a cup first (total cost of half a dozen plates sold at $6000 each - $1200) and use that."""
  print(task_iterate(wrong_soln, feedback))

if __name__ == "__main__":
  test()