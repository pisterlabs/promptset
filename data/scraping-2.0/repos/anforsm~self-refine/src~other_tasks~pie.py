from openai_wrapper import call_openai
import pandas as pd
from tqdm import tqdm
import json

# task_init: PieInit(prompt_examples="data/prompt/pie/init.txt")
# task_feedback: PieFeedback(prompt_examples="data/prompt/pie/feedback.txt")
# task_iterate: PieIterate(prompt_examples="data/prompt/pie/iterate.txt")


# prompt
# question prefix
# slow code
# intra example sep: \n\n
# answer prefix: # Why is this code slow?\n


# first run:
#   task init
#   task feedback
# next:
#   task iterate
#   task feedback

def init(slow_code):
  question_prefix = "# slower version:\n"
  answer_prefix = "# optimized version of the same code:\n"
  intra_example_sep="\n\n\n"
  inter_example_sep="\n\n### END ###\n\n"

  with open("data/init.txt", "r") as f:
    prompt = f.read()
  
  query = f"{prompt}{question_prefix}{slow_code}{intra_example_sep}{answer_prefix}"

  return call_openai(query), query


def iterate(slow_code, feedback):
  question_prefix = ""
  answer_prefix = "# Improved version:\n"
  intra_example_sep="\n\n"
  inter_example_sep = "\n\n### END ###\n\n"

  with open("data/iterate.txt", "r") as f:
    prompt = f.read()
  
  query = instr = "# Why is this code slow?"
  example_template = f"""{slow_code}

{instr}

{feedback}

# Improved version:

"""
  query = f"{prompt}{example_template}"

  return call_openai(query), query

def feedback(slow_code):
  question_prefix = ""
  answer_prefix = "# Why is this code slow?\n"
  intra_example_sep="\n\n"
  inter_example_sep = "\n\n### END ###\n\n"

  with open("data/feedback.txt", "r") as f:
    prompt = f.read()
    
  query = f"""{question_prefix}{slow_code}{intra_example_sep}{answer_prefix}{slow_code}"""

  generated_feedback = call_openai(query)
  return generated_feedback, query
  

def self_refine(slow_code):
  log = []

  for curr_attempt in range(4):
    if curr_attempt == 0:
      fast_code, _ = init(slow_code)
    else:
      fast_code, _ = iterate(slow_code, generated_feedback)
    
    generated_feedback, _ = feedback(fast_code)

    slow_code = fast_code
    log.append({
      "fast_code": fast_code,
      "generated_feedback": generated_feedback,
      "attempt": curr_attempt,
    })
  
  return log

def main():
  programs = pd.read_json("data/code_samples/codenet-python-test-1k.jsonl", lines=True, orient="records")
  results = []
  processed_programs = set()
  for i, row in tqdm(programs.iterrows(), total=len(programs)):
    id = row["submission_id_v0"]
    if id in processed_programs:
      continue
    if i > 10:
      break
    

    processed_programs.add(id)
    slow_code = row["input"]

    result = self_refine(slow_code)

    results.append({
      "id": id,
      "slow_code": slow_code,
      "log": result,
    })
  
  json.dump(results, open("results.json", "w"), indent=2)

if __name__ == "__main__":
  main()