import tokenize
from io import BytesIO
import ast

from openai_wrapper import call_openai
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

import argparse

from multiprocessing import Pool


from readability_prompts_llama import COUNT_VAR_PROMPT, PROMPT_CRITIQUE, PROMPT_FIX


def count_comments(code):
  comment_count = 0
  total_lines = len([l for l in code.splitlines() if l.strip()])

  tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
  for token in tokens:
    if token.type == tokenize.COMMENT:
      comment_count += 1

  return comment_count, comment_count / total_lines

def count_functions(code):
  tree = ast.parse(code)
  return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])

def count_variables(code):
  prompt = COUNT_VAR_PROMPT.format(code=code)
  result = call_openai(prompt)
  result = result.strip().splitlines()
  num_vars = len(result)
  num_random_vars = len([r for r in result if r.endswith("- random")])
  num_meaningful_vars = num_vars - num_random_vars

  return num_meaningful_vars, num_meaningful_vars / num_vars, result

def rererun_failed(model_name):
  fn = model_name.replace("/", "_")
  with open(f"results/{fn}_evaluated.json", "r") as f:
    results = json.load(f)
  
  failed_indices = []
  for i, result in enumerate(results):
    for it in range(len(result["log"])):
      if "failed" in result["log"][it] and result["log"][it]["failed"]:
        failed_indices.append(i)
        break
  
  def extract_code(s):
    start = ":\n\n"
    if start in s:
      s= s[s.find(start) + len(start):]

    start = "\n\n\n"
    if start in s:
      s = s[:s.find(start)]

    if "```" not in s:
      return s

    start = "```"

    left = s.find(start)
    right = s.rfind(start)

    if left == right:
      print("found super special case")
      return s[:right]

    s = s[s.find(start) + len(start):s.rfind(start)]

    if "```" in s:
      s = s.replace("```", "")
    return s.strip()

  still_failed = 0
  print(f"Rerunning {len(failed_indices)} failed programs")
  for i in tqdm(failed_indices):
    result = results[i]
    old_code = None
    for it in range(len(result["log"])):
      try:
        if old_code is None:
          try:
            old_code = result["log"][it]["old_code"]
            old_comment_count, old_comment_density = count_comments(old_code)
            old_num_functions = count_functions(old_code)
            old_num_meaningful_vars, old_var_density, old_vars = count_variables(old_code)
          except:
            old_code = extract_code(result["log"][it]["old_code"])
            old_comment_count, old_comment_density = count_comments(old_code)
            old_num_functions = count_functions(old_code)
            old_num_meaningful_vars, old_var_density, old_vars = count_variables(old_code)

        result["log"][it].update({
          "old_comment_count": old_comment_count,
          "old_comment_density": old_comment_density,
          "old_num_functions": old_num_functions,
          "old_num_meaningful_vars": old_num_meaningful_vars,
          "old_var_density": old_var_density,
          "old_vars": old_vars,
        })
    
        try:
          new_code = result["log"][it]["new_code"]
          comment_count, comment_density = count_comments(new_code)
          num_functions = count_functions(new_code)
          num_meaningful_vars, var_density, vars = count_variables(new_code)
        except:
          new_code = extract_code(result["log"][it]["new_code"])
          comment_count, comment_density = count_comments(new_code)                   
          num_functions = count_functions(new_code)                                   
          num_meaningful_vars, var_density, vars = count_variables(new_code)

        result["log"][it].update({
          "comment_count": comment_count,
          "comment_density": comment_density,
          "num_functions": num_functions,
          "num_meaningful_vars": num_meaningful_vars,
          "var_density": var_density,
          "vars": vars,
          "failed": False,
        })

        old_code = new_code
        old_comment_count = comment_count
        old_comment_density = comment_density
        old_num_functions = num_functions
        old_num_meaningful_vars = num_meaningful_vars
        old_var_density = var_density
        old_vars = vars
      except Exception as e:
        print(e)

        still_failed += 1
        result["log"][it].update({
          "error_message": str(e),
        })

        for it2 in range(it, len(result["log"])):
          result["log"][it2].update({
            "failed": True,
          })

        break

  print(f"Still failed: {still_failed}")
  json.dump(results, open(f"results/{fn}_evaluated.json", "w"), indent=2)


def rerun_failed(model_name):
  fn = model_name.replace("/", "_")
  with open(f"results/{fn}_evaluated.json", "r") as f:
    results = json.load(f)
  
  failed_indices = []
  for i, result in enumerate(results):
    for it in range(len(result["log"])):
      if "failed" in result["log"][it] and result["log"][it]["failed"]:
        failed_indices.append(i)
        break
  
  def extract_code(s):
    if "```" not in s:
      return s
    
    s = s.replace("```python", "```")

    start = "```"

    left = s.find(start)
    right = s.rfind(start)

    if left == right:
      if left < len(s) / 2:
        return s[left + len(start):]
      else:
        return s[:right]

    return s[s.find(start) + len(start):s.rfind(start)]

  still_failed = 0
  print(f"Rerunning {len(failed_indices)} failed programs")
  for i in tqdm(failed_indices):
    result = results[i]
    old_code = None
    for it in range(len(result["log"])):
      try:
        if old_code is None:
          old_code = extract_code(result["log"][it]["old_code"])
          old_comment_count, old_comment_density = count_comments(old_code)
          old_num_functions = count_functions(old_code)
          old_num_meaningful_vars, old_var_density, old_vars = count_variables(old_code)

        result["log"][it].update({
          "old_comment_count": old_comment_count,
          "old_comment_density": old_comment_density,
          "old_num_functions": old_num_functions,
          "old_num_meaningful_vars": old_num_meaningful_vars,
          "old_var_density": old_var_density,
          "old_vars": old_vars,
        })
    
        new_code = extract_code(result["log"][it]["new_code"])
        comment_count, comment_density = count_comments(new_code)
        num_functions = count_functions(new_code)
        num_meaningful_vars, var_density, vars = count_variables(new_code)

        result["log"][it].update({
          "comment_count": comment_count,
          "comment_density": comment_density,
          "num_functions": num_functions,
          "num_meaningful_vars": num_meaningful_vars,
          "var_density": var_density,
          "vars": vars,
          "failed": False,
        })

        old_code = new_code
        old_comment_count = comment_count
        old_comment_density = comment_density
        old_num_functions = num_functions
        old_num_meaningful_vars = num_meaningful_vars
        old_var_density = var_density
        old_vars = vars
      except Exception as e:

        still_failed += 1
        result["log"][it].update({
          "error_message": str(e),
        })

        for it2 in range(it, len(result["log"])):
          result["log"][it2].update({
            "failed": True,
          })

        break

  print(f"Still failed: {still_failed}")
  json.dump(results, open(f"results/{fn}_evaluated.json", "w"), indent=2)

def run_eval(model_name):
  fn = model_name.replace("/", "_")
  with open(f"results/{fn}results.json", "r") as f:
    results = json.load(f)
  
  evaled_indices = []
  max_num_evals = 50
  num_evals = 0
  remove_indices = []
  def extract_code(s):
    if "```" not in s:
      return s
    
    return s.rstrip("```")

  for i, result in tqdm(enumerate(results), total=max_num_evals):
    old_code = None
    for it in range(len(result["log"])):
      try:
        if old_code is None:
          old_code = extract_code(result["log"][it]["old_code"])
          old_comment_count, old_comment_density = count_comments(old_code)
          old_num_functions = count_functions(old_code)
          old_num_meaningful_vars, old_var_density, old_vars = count_variables(old_code)

        result["log"][it].update({
          "old_comment_count": old_comment_count,
          "old_comment_density": old_comment_density,
          "old_num_functions": old_num_functions,
          "old_num_meaningful_vars": old_num_meaningful_vars,
          "old_var_density": old_var_density,
          "old_vars": old_vars,
        })
    
        new_code = extract_code(result["log"][it]["new_code"])
        comment_count, comment_density = count_comments(new_code)
        num_functions = count_functions(new_code)
        num_meaningful_vars, var_density, vars = count_variables(new_code)

        result["log"][it].update({
          "comment_count": comment_count,
          "comment_density": comment_density,
          "num_functions": num_functions,
          "num_meaningful_vars": num_meaningful_vars,
          "var_density": var_density,
          "vars": vars,
          "failed": False,
        })

        old_code = new_code
        old_comment_count = comment_count
        old_comment_density = comment_density
        old_num_functions = num_functions
        old_num_meaningful_vars = num_meaningful_vars
        old_var_density = var_density
        old_vars = vars
      except Exception as e:

        result["log"][it].update({
          "error_message": str(e),
        })

        for it2 in range(it, len(result["log"])):
          result["log"][it2].update({
            "failed": True,
          })

        break


    num_evals += 1
    evaled_indices.append(i)
    if num_evals >= max_num_evals:
      break
  
  #for i in remove_indices[::-1]:
  #  results.pop(i)
  #for i in range(len(results)):
  #  if i not in evaled_indices:
  #    results.pop(i)
  new_results = []
  for i in evaled_indices:
    new_results.append(results[i])
  
  json.dump(new_results, open(f"results/{fn}_evaluated.json", "w"), indent=2)

def draw_graphs(model_name):
  fn = model_name.replace("/", "_")
  results = json.load(open(f"results/{fn}_evaluated.json", "r"))
  comment_ratios = []
  var_ratios = []
  func_nums = []
  comment_ratios.append(sum([r["log"][0]["old_comment_density"] for r in results]) / len(results))
  var_ratios.append(sum([r["log"][0]["old_var_density"] for r in results]) / len(results))
  func_nums.append(sum([r["log"][0]["old_num_functions"] for r in results]) / len(results))
  for it in range(len(results[0]["log"])):
    comment_ratios.append(sum([r["log"][it]["comment_density"] for r in results]) / len(results))
    var_ratios.append(sum([r["log"][it]["var_density"] for r in results]) / len(results))
    func_nums.append(sum([r["log"][it]["num_functions"] for r in results]) / len(results))
  
  plt.plot(comment_ratios)
  plt.plot(var_ratios)
  plt.plot(func_nums)
  plt.legend(["Comment ratio", "Variable ratio", "Function number"])
  plt.show()

def calculate_stats(model_name):
  fn = model_name.replace("/", "_")
  with open(f"results/{fn}_evaluated.json", "r") as f:
    results = json.load(f)

  avg_comment_ratio = 0
  avg_func_num = 0
  avg_var_ratio = 0

  old_avg_comment_ratio = 0
  old_avg_func_num = 0
  old_avg_var_ratio = 0

  for i, result in enumerate(results):
    old_avg_comment_ratio += result["log"][0]["old_comment_density"]
    old_avg_func_num += result["log"][0]["old_num_functions"]
    old_avg_var_ratio += result["log"][0]["old_var_density"]
    
    avg_comment_ratio += result["log"][-1]["old_comment_density"]
    avg_func_num += result["log"][-1]["old_num_functions"]
    avg_var_ratio += result["log"][-1]["old_var_density"]
    

  avg_comment_ratio /= len(results)
  avg_func_num /= len(results)
  avg_var_ratio /= len(results)
  
  old_avg_comment_ratio /= len(results)
  old_avg_func_num /= len(results)
  old_avg_var_ratio /= len(results)
  
  print(f"Average comment ratio: {old_avg_comment_ratio} -> {avg_comment_ratio}")
  print(f"Average function number: {old_avg_func_num} -> {avg_func_num}")
  print(f"Average variable ratio: {old_avg_var_ratio} -> {avg_var_ratio}")





def main(model_name):
  import time
  #run_eval(model_name)
  #time.sleep(10)
  #rerun_failed(model_name)
  #time.sleep(10)
  rererun_failed(model_name)
  #calculate_stats(model_name)
  #draw_graphs(model_name)


if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--model", type=str, required=True)
  args = argparser.parse_args()
  model_name = args.model
  main(model_name)