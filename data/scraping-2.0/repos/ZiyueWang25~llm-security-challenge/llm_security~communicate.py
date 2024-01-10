import time
import re
from tqdm import tqdm

from pwn import ssh

import pandas as pd
import numpy as np
import openai
import replicate

from llm_security import llm_config 
from llm_security import game_config 


def find_response(output):
  splits = output.strip().split("\n")
  for i, l in enumerate(splits):
    if l.lstrip().startswith("#"):
      continue
    else:
      return "\n".join(splits[:i+1])
  return output


def get_gpt_response(model, messages):
  response = openai.ChatCompletion.create(messages=messages, model=model, max_tokens=100)
  output = response.choices[0].message.content
  llm_response = find_response(output)
  return {"role": "assistant", "content": llm_response}, llm_response


def get_gpt_comp_response(model, messages):
  response = openai.Completion.create(prompt="\n".join(messages), model=model, max_tokens=100)
  output = response.choices[0].text
  llm_response = find_response(output)
  return llm_response, llm_response


def get_replicate_response(model, messages, system_prompt):
  output = "".join(replicate.run(
    model,
    input={
      "system_prompt": system_prompt,
      "prompt": "\n".join(messages),
    }
  ))
  llm_response = find_response(output)
  return llm_response, llm_response


def get_model_response(model, messages, sys_content=None):
  start_time = time.time()
  abbr = llm_config.ABBR_BY_MODEL_NAME[model]
  if abbr.startswith(llm_config.OPENAI_LIST):
    response, llm_response = get_gpt_response(model, messages)
  elif abbr.startswith(llm_config.OPENAI_COMP_LIST):
    response, llm_response = get_gpt_comp_response(model, messages)
  elif abbr.startswith(llm_config.REPLICATE_LIST):
    response, llm_response = get_replicate_response(model, messages, sys_content)
  elapse_secs = time.time() - start_time
  return response, llm_response, elapse_secs


def get_initial_messages(model, sys_content):
  abbr = llm_config.ABBR_BY_MODEL_NAME[model]
  if abbr.startswith(llm_config.OPENAI_LIST):
    messages = [
      {'role': 'system', 'content': sys_content},
    ]
  elif abbr.startswith(llm_config.OPENAI_COMP_LIST):
    messages = [sys_content, "Let's code:"]
  elif abbr.startswith(llm_config.REPLICATE_LIST):
    messages = ["Let's code:"]
  return messages


def get_ssh_arg(level, password, game):
  return dict(
    user = f"{game}{level}",
    host = f"{game}.labs.overthewire.org",
    password = password,
    port=game_config.PORT_BY_GAME[game],
  )  

  
def get_ssh_response(io, llm_response, sleep_time, max_len=1000):
  io.sendlines([i.encode() for i in llm_response.split('\n')])
  time.sleep(sleep_time)
  ssh_response = safe_decode(io.recv())
  
  if len(ssh_response) > max_len:
    ssh_response = "Too long"
  if not ssh_response:
    ssh_response = 'Empty'
  ssh_response = "SHELL:" + ssh_response
  return ssh_response


def safe_decode(b):
  try:
    decode = b.decode()
  except:
    decode = str(b)[2:-1]
  decode = decode.replace("$ ", "") # it looks like the ssh generate message like "$ $ $ "
  return decode


def make_response_df(llm_rs, ssh_rs, secs):
  len_diff = len(llm_rs)-len(ssh_rs)
  if len_diff > 0:
    ssh_rs.extend([""] *len_diff)
  elif len_diff < 0:
    llm_rs.extend([""] * -len_diff)    
    secs.extend([0] * -len_diff)
  return pd.DataFrame({"llm": llm_rs, "ssh": ssh_rs, "sec": secs})


def is_answer_correct(game, level, answer):
  if answer is None:
    return False
  return answer == game_config.PSW_BY_GAME[game][level+1]
  
def find_answer(llm_response, verbose=True):
  if m := re.findall(r'(?:^|\n)submit (\w+)', llm_response):
    answer = m[-1].strip()
    if verbose:
      print('SYS: we are done. Predicted Answer: ', answer)
    return True, answer
  return False, None


def attempt_level(
  game,
  level,
  n_cycles,
  model="gpt-3.5-turbo",
  password=None,
  verbose=True,
  sleep_time=.5,
  session_timeout=30,
  max_ssh_len = 1000,
): 
  if verbose:
    print(f"The max ssh_len is: {max_ssh_len}")
  abbr = llm_config.ABBR_BY_MODEL_NAME[model]
  goal_by_level = game_config.GOAL_BY_GAME[game]
  sys_goal_suffix = f"The goal is: {goal_by_level[level+1]}." if isinstance(goal_by_level, dict) else goal_by_level
  if verbose:
    print(f"Level: {level}. Goal: {sys_goal_suffix}")
  sys_content = llm_config.PRELUDE + sys_goal_suffix
  messages = get_initial_messages(model, sys_content)

  llm_responses = []
  ssh_responses = []
  secs = [] 
  answer = None
  if password is None:
    password = game_config.PSW_BY_GAME[game][level]
  ssh_args = get_ssh_arg(level, password, game)
  with ssh(raw=True, timeout=session_timeout, **ssh_args,) as session:
    with session.process(['sh']) as io:
      for i in range(n_cycles):
        try:
          response, llm_response, sec = get_model_response(model, messages, sys_content)
          messages.append(response)
          llm_responses.append(llm_response)
          secs.append(sec)
          if verbose:
            print(f'SYS {i}: LLM: \n>>>{llm_response}<<<\n')
        except openai.error.InvalidRequestError as err:
          print(f"Maximum context length reached with error message: {err}")
          messages = messages[:4] + messages[-4:]
          llm_response = ""

        found, answer = find_answer(llm_response, verbose=verbose)
        if found:
          break
        try:
          ssh_response = get_ssh_response(io, llm_response, sleep_time, max_ssh_len)  
        except EOFError:
          break
          
        if verbose:
          print(f'SYS {i}: ssh: \n>>>{ssh_response}<<<\n')

        if abbr.startswith(llm_config.OPENAI_LIST):
          messages.append({'role': 'user', 'content': ssh_response})
        elif abbr.startswith(llm_config.REPLICATE_LIST) or abbr.startswith(llm_config.OPENAI_COMP_LIST):
          messages.append(ssh_response)
        ssh_responses.append(ssh_response)
  return make_response_df(llm_responses, ssh_responses, secs), answer


def attempt_level_multiple(game, model_name, level, num_trial=100):
  logs = []
  print(f"---------- Test Model {model_name} ------------")
  for _ in tqdm(range(num_trial)):
    try:
      resp_df, answer = attempt_level(game, level, model=model_name, n_cycles=30, verbose=False)
    except Exception as e:
      print(f"Error: {e}")
      logs.append((None, None, e))
      continue
    print(len(resp_df), answer)
    logs.append((resp_df, answer, None))
  return logs


def show_level_trials(abbrs, game, level, num_trial, logs_by_model):
  print(f"Under {num_trial} trials, In level {level}")
  for abbr in abbrs:
    model = llm_config.MODEL_NAME_BY_ABBR[abbr]
    if model not in logs_by_model:
      continue
    print("#"*10 + f" {abbr} " + "#"*10)
    num_success = sum([is_answer_correct(game, level, l[1] ) for l in logs_by_model[model]])
    rate = num_success / num_trial
    attempts = [l[0].shape[0] if l[0] is not None else num_trial for l in logs_by_model[model]]
    avg_attempts, std_attempts = np.mean(attempts), np.std(attempts)
    print(f"model {model:15s}, success rate: {rate:.1%}, avg turns to solve the level: {avg_attempts:.1f}, std: {std_attempts:.1f}")
