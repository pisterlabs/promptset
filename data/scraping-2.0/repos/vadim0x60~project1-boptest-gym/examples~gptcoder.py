# For Program Synthesis
import openai
import programlib
from pexpect.exceptions import EOF, TIMEOUT

# Dealing with OpenAI API requires tenacity
from tenacity import retry, retry_if_exception_message, retry_if_exception_type, wait_random_exponential, stop_after_attempt, retry_if_result

# For BOPTEST
import gymnasium as gym
from boptestGymEnv import BoptestGymEnv
from examples.test_and_plot import test_agent, retreive_results

import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import sys
import itertools


START_STR = 'INITIATE SYNTHESIZE, EXECUTE, INSTRUCT, DEBUG'

model = 'gpt-4'
openai_tantrums = (openai.error.RateLimitError, 
                   openai.error.APIError, 
                   openai.error.ServiceUnavailableError)

SYSTEM_MSG = os.environ.get('SYSTEM_MSG', 'You are a program synthesis system. Answer with code only.')
SUMMARY_FACTOR = int(os.environ.get('SUMMARY_FACTOR', 100))
N = int(os.environ.get('N', 4096))

print(START_STR)
for config_var in ('SYSTEM_MSG', 'SUMMARY_FACTOR', 'N'):
  print(f'{config_var} = {eval(config_var)}')

def log_last():
  """Log the last message to stdout"""
  print('---')
  print(messages[-1]['content'], flush=True)

def log_all():
  """Log all messages except the system prompt to stdout"""
  for msg in messages[1:]:
    print('---')
    print(msg['content'], flush=True)

def restore_state():
  if sys.stdin.isatty():
    raise ValueError('no piped input found')
  
  message_archive = sys.stdin.read().split('\n---\n')
  message_archive = filter(lambda x: START_STR not in x, message_archive)
  roles = itertools.cycle(['user', 'assistant'])
  message_archive = [
    {'role': role, 'content': msg} 
    for role, msg in zip(roles, message_archive)
  ]

  if (message_archive[-1]['role'] == 'assistant'):
    message_archive.pop()

  messages.extend(message_archive)

# Make sure that the test case is bestest_hydronic_heat_pump
# TESTCASE=bestest_hydronic_heat_pump docker compose up
url = 'http://127.0.0.1:5000'

observations = OrderedDict([('reaTZon_y', (280.0, 310.0)),
                            ('PriceElectricPowerDynamic', (0, 10)), 
                            ('TDryBul', (200, 400))])

observations_hint = """
- reaTZon_y: the zone temperature in Kelvin
- PriceElectricPowerDynamic[49]: the forecasted electricity price in Euro per kWh, in 15-minute intervals
- TDryBul[49]: the forecasted dry bulb temperature outside in Kelvin, in 15-minute intervals

99 variables in total, each on a new line.
"""

report_observations = ['reaTZon_y', 'oveHeaPumY_u', 'reaPHeaPum_y', 
                       'reaTSup_y', 'reaTRet_y']

report_observations_hint = """
- reaTZon_y: the zone temperature in Kelvin
- oveHeaPumY_u: the heat pump modulating signal (0-1)
- reaPHeaPum_y: Heat pump electrical power
- reaTSup_y: Supply water temperature to radiant floor
- reaTRet_y: Return water temperature from radiant floor
"""

env = BoptestGymEnv(
  url                   = url,
  actions               = ['oveHeaPumY_u'],
  observations          = observations,
  random_start_time     = False,
  predictive_period     = 12 * 3600,
  start_time            = 1382400,
  warmup_period         = 7 * 24 * 3600,
  max_episode_length    = 7 * 24 * 3600,
  step_period           = 900,
  scenario              = {'electricity_price': 'dynamic', 'time_period': 'peak_heat_day'})

def brief():
  # bestest_hydronic_heat_pump test case,
  
  text = """
    Write a program to control a Hydronic Heat Pump in a simplified residential dwelling for a family of 5 members, modeled as a single thermal zone, located in Brussels, Belgium. The building envelope model is based on the BESTEST case 900 test case. but it is scaled to an area that is four times larger. The rectangular floor plan is 12 m by 16 m. Internal walls are configured such that there are around 12 rooms in the building. The builiding further contains 24 m2 of windows on the south facade.

    An air-to-water modulating heat pump of 15 kW nominal heating capacity extracts energy from the ambient air to heat up the floor heating emission system. A fan blows ambient air through the heat pump evaporator and circulation pump pumps water from the heat pump to the floorr when the heat pump is operating. 
    
    The program should be an infinite loop that reads input variables with input() and outputs the heat pump modulating signal (oveHeaPumY_u) for compressor speed between 0 (not working) and 1 (working at maximum capacity) with print(). There should be no output other then the control signal. The program should be written in Python.
    
    Input variables are, in this order:"""
  text += observations_hint
  messages.append({'role': 'user', 'content': text})

def summarize(df):
  text = """
    Below you will find a rollout of the Hydronic Heat Pump environment.
    It represents a history of one thermostat control episode
    Recorded variables are:"""
  
  text += report_observations_hint
  
  text += '\n' + df.to_string(float_format=lambda x: f'{x:.4f}') + '\n'

  text += """

    Can you write a short summary of what happened?
    """
  
  messages = [
    {'role': 'system', 'content': 'You are a helpful assistant'},
    {'role': 'user', 'content': text}
  ]

  gpt(messages)()
  return messages[-1]['content']

def debrief(rollout, kpis):
  text = """
    Here's how it went:
    """
  
  # Resample the rollout to reduce the number of tokens for GPT to handle
  episode_length = len(rollout)
  episode_days = (rollout.iloc[-1]['time'] - rollout.iloc[0]['time']) / (3600 * 24)
  if episode_length > SUMMARY_FACTOR:
    groups = np.arange(episode_length) // int (episode_length / SUMMARY_FACTOR)
    rollout = rollout.groupby(groups).mean()
  text += summarize(rollout)

  text += f"""
    Episode length: {episode_days} days

    Daily electrcity cost: {kpis['cost_tot'] / episode_days} EUR/m2
    Daily thermal discomfort: {kpis['tdis_tot'] / episode_days} K*h/zone
    Daily energy use (kWh): {kpis['ener_tot'] / episode_days} kWh/m2
    Daily emissions: {kpis['emis_tot'] / episode_days} kgCO2/m2

    Computational time ratio: {kpis['time_rat']}

    Can you rewrite the program to lower the costs and/or discomfort?
    """
  messages.append({'role': 'user', 'content': text})

messages = [
    {'role': 'system', 'content': SYSTEM_MSG},
]

def gpt(messages):
  def prune_messages(*args):
    # message[0] is the system message
    del messages[1]

  @retry(retry=retry_if_exception_message(match=r'.*Please reduce the length.*'),
         after=prune_messages)
  @retry(retry=retry_if_result(lambda r: r['finish_reason'] == 'length'),
         after=prune_messages)
  @retry(retry=retry_if_exception_type(openai_tantrums),
         wait=wait_random_exponential(),
         stop=stop_after_attempt(10))
  def infer():
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1
      )
    return completion['choices'][0]
  
  def continue_chat():
    messages.append(infer()['message'])

  return continue_chat

def extract_code(msg):
  if '```' in msg:
    msg = msg.split('```')[1]
    msg = msg.split('\n')[1:]
    return '\n'.join(msg)
  else:
    return msg

def test(episode_length):
  code = extract_code(messages[-1]['content'])
  program = programlib.Program(code, language='Python')
  model = program.spawn().rl(env.action_space, env.observation_space)
  
  test_agent(env, model, env.start_time, episode_length, env.warmup_period)
  rollout = retreive_results(env, points=report_observations)
  rollout = rollout[rollout['time'] > env.start_time]
  rollout = rollout[rollout['time'] < env.start_time + env.max_episode_length]

  return rollout, env.get_kpis()

try:
  restore_state()
except ValueError:
  brief()
log_all()

episode_length = min(32 * 60 * (2 ** ((len(messages) - 1) / 2)), 7 * 24 * 3600)

for i in range(N):
  gpt(messages)()
    
  log_last()

  try:
    rollout, kpis = test(episode_length)
    debrief(rollout, kpis)
    episode_length = min(episode_length * 2, 7 * 24 * 3600)
  except ValueError as e:
    messages.append({'role': 'user', 'content': str(e)})
  except (OSError, EOF, TIMEOUT) as e:
    messages.append({'role': 'user', 'content': 'Your program doesn\'t seem to be expecting input.'})
  except SyntaxError as e:
    messages.append({'role': 'user', 'content': 'Are you sure your program outputs only the control signal (a number)?'})

  log_last()