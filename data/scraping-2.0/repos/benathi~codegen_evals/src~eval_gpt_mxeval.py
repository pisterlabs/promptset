import logging
import openai
from mxeval.evaluation import (
  get_execute_function,
  estimate_pass_at_k,
)
from mxeval.data import get_data, get_examples, get_supported_langs
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
import numpy as np
import time
import argparse
import sys
import datetime
import wandb


## YOUR API KEY HERE
openai.api_key_path = ".openai_key"

def print_teal(text):
    teal_text = "\x1b[36m" + text + "\x1b[0m"
    print(teal_text)

def print_lavender(text):
  lavender_text = "\x1b[94m" + text + "\x1b[0m"
  print(lavender_text)

def print_pink(text):
  pink_text = "\x1b[95m" + text + "\x1b[0m"
  print(pink_text)


def get_args():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                          choices=["gpt-3.5-turbo", "gpt-4"])
  arg_parser.add_argument("--dataset", type=str, default="multi-humaneval",
                          choices=["multi-humaneval", "mbxp", "mathqa-x"]
                          )
  arg_parser.add_argument("--num_shots", type=int, default=1,
                          )
  arg_parser.add_argument("--num_turns", type=int, default=1,
                          )
  arg_parser.add_argument("--verbose", type=int, default=0,
                          )
  arg_parser.add_argument("--language", type=str, default="all")
  arg_parser.add_argument("--language_exclude", type=str, default=None)
  arg_parser.add_argument("--temp", type=float, default=1.0)
  arg_parser.add_argument("--experiment_name", type=str, default=None)
  arg_parser.add_argument("--use_execution_feedback", type=int, default=0)
  arg_parser.add_argument("--use_wandb", type=int, default=0)

  arg_parser.add_argument("--system_message_suggest_test", type=int, default=0)
  ## for debugging -- number of problems
  arg_parser.add_argument("--log_level", type=str, default="INFO")
  arg_parser.add_argument("--limit_num_problems", type=int, default=None,
                          help="If not none, this is a debug mode where the evaluation happens only on a subset of problems.")
  args = arg_parser.parse_args()
  return args


def construct_messages(turn_idx,
                       problem,
                       execution_result=None,
                       previous_response=None,
                       # messages=None,
                       fewshot_examples=[]):
  """
    This code is used to construct the message history + next turn's prompt,
    which is to be passed to the model at the next stage.

    Future refactoring will use langchain.
  """
  if args.system_message_suggest_test:
    execute_message = "Feel free to generate your own test cases after completing the function. "


  if turn_idx == 0:
    # first turn: construct messages
    # (1) high level instruction
    new_messages = [
      {"role": "system",
       "content": "You are an expert coder in all programming languages. Please continue writing code based on each function signature without repeat the function signature. If you write any explanations in English sentences, please wrap them in comments in the correct format according to that language. " + execute_message},
    ]
    # (2) examples
    for fs_example in fewshot_examples:
      new_messages.append({"role": "user", "content": fs_example["prompt"]})
      new_messages.append({"role": "assistant", "content": fs_example["completion"]})
    # (3) add the actual prompt
    prompt = problem["prompt"]
    new_messages.append({"role": "user", "content": prompt})
  else:
    new_messages = []
    # add (1) model's previous response
    assert previous_response is not None, "For next turn, we require providing the previous assistant's response"
    new_messages.append({"role": "assistant", "content": previous_response})
    # and (2) execution feedback.
    prompt = ""
    if args.use_execution_feedback:
      prompt += "Below is the error message\n-----------------------\n" \
      + execution_result \
      + f"\n-----------------------"
    prompt += "\nPlease solve the following problem again\n"
    prompt += problem["prompt"]
    # second turn: append user and assistant messages
    new_messages.append({"role": "user", "content": prompt})

  return new_messages, prompt


def query_code_completion(model_name,
                          messages,
                          verbose=False,
                          kwargs={},
                          retry_count=5):
  # calling API
  logging.debug("Initiating the API call")
  request_result = None
  for i in range(retry_count):
    logging.debug(f"API Call number {i + 1}")
    try:
      request_result = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        timeout=1.0,
        **kwargs,
      )
      break
    except Exception as e:
      logging.warning(e)
      time.sleep(1)
  if request_result is None:
    response = ""
    logging.warning(f"Warning!! -- API call failed after {retry_count} retries")
  else:
    if verbose:
      logging.debug(request_result)
    response = request_result["choices"][0]["message"]["content"]

  return response, messages, request_result  # code, request_result


def wrapper_pass_at_k(k_list, total, correct):
  """
  Total:   [num attempts problem 1, num attempts problem 2, ...]
  Correct: [num correct problem 1, num correct problem 2, ...]
  """
  total = np.array(total)
  correct = np.array(correct)
  pass_at_k = {
    f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
    for k in k_list
    if (total >= k).all()
  }
  return pass_at_k


def display_messages(messages):
  """
    return a string representation of messages (list of role & content)
  """
  s = ""
  for message in messages:
    s += f"{message['role']}:\n{message['content']}\n"
  return s


def eval_language(dataset="mbxp",
                  model_name="gpt-3.5-turbo",
                  language="javascript",
                  num_turns=0,
                  n_workers=1,
                  verbose=False,
                  # answer_full_function=False,
                  execute=True,
                  num_reps=1,
                  limit_num_problems=None,
                  k_shot=2,
                  temperature=0):
  check_correctness_function = get_execute_function(language)
  data_obj = get_data(dataset=dataset, language=language)  # specific language or all
  # TODO -- change this to huggingface loader for convenience
  fewshot_examples = get_examples(dataset="mbxp",  # other datasets have similar few shot examples as mbxp
                                  language=language,
                                  num_examples=k_shot)
  total, correct = {}, {}
  num_turns_used = []
  for idx, task_id in enumerate(data_obj):
    if limit_num_problems and idx == limit_num_problems:
      break
    problem = data_obj[task_id]
    passes = []
    for rep in range(num_reps):
      messages, previous_response, execution_result = [], None, None
      num_turns_used.append(0)
      for turn_idx in range(num_turns):
        num_turns_used[-1] += 1
        new_messages, prompt = construct_messages(turn_idx,
                                                  problem,
                                                  execution_result,
                                                  previous_response=previous_response,
                                                  fewshot_examples=fewshot_examples)
        messages += new_messages

        response, messages, _ = query_code_completion(
          model_name=model_name,
          messages=messages,
          verbose=False,
          kwargs={"temperature": temperature},
        )

        code = response
        s = "\n\n"
        s += (f"##################### Task ID {task_id} | Turn {turn_idx}/{num_turns} (0 indexed) #############\n")
        s += display_messages(new_messages)
        s2 = ""
        s2 += ("\n-----------------[ completion ]------------------\n")
        s2 += (code)
        if verbose > 2:
          print_lavender(s)
          print_teal(s2)
        kwargs = {"problem": problem,
                  "completion": code,
                  "timeout": 30.0,
                  "completion_id": 0,
                  }
        if execute:
          with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future = executor.submit(check_correctness_function, **kwargs)
            result = future.result()
            passed = result["passed"]
            execution_result = result["result"]

            if verbose > 2:
              s = (f"********** execution result | passed = {passed} ***********\n")
              s += (execution_result)
              print_pink(s)
            if passed:
              break  # no need to keep asking in multi turn
            else:
              previous_response = code

      passes.append(passed)
    if execute:
      assert len(
        passes) == num_reps, f"Number of passes does not match number of attempts -- {len(passes)} != {num_reps} -- {passes}"
      total[task_id] = num_reps
      correct[task_id] = np.sum(passes)

  # after loop through everything, calculate scores
  if execute:
    # prepare total and correct list
    total_list, correct_list = [], []
    for task_id in total:
      total_list.append(total[task_id])
      correct_list.append(correct[task_id])
    pass_at_k = wrapper_pass_at_k(k_list=[1, num_reps], total=total_list, correct=correct_list)
    logging.info(pass_at_k)
    logging.info("num turns used")
    logging.info(num_turns_used)

    num_turns_used_array = np.array(num_turns_used)
    stats_dict = {
      "mean": np.mean(num_turns_used_array),
      "std_deviation": np.std(num_turns_used_array),
      "median": np.median(num_turns_used_array),
      "min": np.min(num_turns_used_array),
      "max": np.max(num_turns_used_array)
    }

    wandb.log({"pass_at_k": pass_at_k,
                "num_turns_used": stats_dict,
                }
              )
    wandb.finish()




def get_log_filename(args, experiment_name=None):
  # Generate experiment name based on args
  experiment_name = experiment_name or "experiment"
  exclude = "_minus" + args.language_exclude if args.language_exclude else ""
  experiment_name += f"_{args.model_name}_{args.dataset}_{args.language}{exclude}_shots{args.num_shots}_turns{args.num_turns}"
  if args.limit_num_problems:
    experiment_name += f"_limit{args.limit_num_problems}"
  if args.temp != 0.2:
    experiment_name += f"_temp{args.temp}"
  if args.verbose > 0:
    experiment_name += f"_verbose{args.verbose}"
  if args.experiment_name:
    experiment_name += f"_{args.experiment_name}"

  # Generate filename with timestamp
  now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  filename = f"logs/{experiment_name}_{now}.log"
  return filename


def eval_all_langs():

  log_level = logging.DEBUG if args.log_level == "debug" else logging.INFO

  # Set up logging
  logging.basicConfig(level=log_level,
                      format='%(asctime)s %(levelname)s %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
  fh = logging.FileHandler(get_log_filename(args))
  fh.setLevel(log_level)
  fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
  logging.getLogger().addHandler(fh)

  for language in get_supported_langs(args.dataset):
    if (args.language != "all" and language != args.language) or \
      (args.language_exclude is not None and language in args.language_exclude):
      continue
    eval_language(dataset=args.dataset,
                  model_name=args.model_name,
                  language=language,
                  verbose=args.verbose,  # 2 for most verbose
                  limit_num_problems=args.limit_num_problems,
                  num_turns=args.num_turns,
                  temperature=args.temp,
                  k_shot=args.num_shots,
                  )
  logging.shutdown()


def setup_wandb():
  # start a new wandb run to track this script
  wandb.init(
    # set the wandb project where this run will be logged
    project="code-agents",

    # track hyperparameters and run metadata
    config=vars(args)
  )


if __name__ == "__main__":
  args = get_args()
  setup_wandb()
  eval_all_langs()
