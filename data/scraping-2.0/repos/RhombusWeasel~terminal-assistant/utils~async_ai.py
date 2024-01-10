import openai, uuid, subprocess
import configparser
import colorama
import json
from utils.logger import Logger
from utils.db import Database
from utils.json_validator import parse_json_string

config = configparser.ConfigParser()
config.read('config.ini')

colorama.init()
openai.api_key = config.get('ai', 'api_key')

class Agent:
  def __init__(self, name_prefix='AI-', log_level=Logger.ERROR, print_tokens=True, model='gpt-3.5-turbo-16k'):
    self.uuid = str(uuid.uuid4())
    self.logger = Logger('{}:{}'.format(name_prefix, self.uuid.split('-')[0]), log_level=log_level)
    self.model = model
    self.show_token_use = print_tokens
    self.db = Database('requests.db')
    self.db.create_table('blackboards')

  def print_tokens(self, response):
    if self.show_token_use:
      usage = response.usage
      breakdown = f'Token Usage: Prompt: {usage["prompt_tokens"]} | AI: {usage["completion_tokens"]} | Total: {usage["total_tokens"]}'
      if usage['total_tokens'] <= 3000:
        col = colorama.Fore.GREEN
      elif usage['total_tokens'] <= 7000:
        col = colorama.Fore.YELLOW
      else:
        col = colorama.Fore.RED
      print(f'{col}{breakdown}{colorama.Style.RESET_ALL}')

  def get_response(self, m, functions=[], function_call='auto', temperature=0.2):
    if len(functions) > 0:
      response = openai.ChatCompletion.create(
        model=self.model,
        messages=m,
        functions=functions,
        function_call=function_call,
        temperature=temperature,
      )
    else:
      response = openai.ChatCompletion.create(
        model=self.model,
        messages=m,
        temperature=temperature,
      )
    self.print_tokens(response)
    txt = {
      "text": response.choices[0]['message'],
      "tokens": response.usage
    }
    return txt

  def process_request(self, req_uuid, p):
    response_text = self.get_response(p)
    self.logger.info(f'Response received: {response_text}')
    self.db.save_data("blackboards", req_uuid, "response", json.dumps(response_text))

  def check_response_status(self, req_uuid):
    response_text = self.db.load_data("blackboards", req_uuid, "response")
    if response_text != '{"uuid": "pending"}':
      return parse_json_string(response_text)
    else:
      return False

  def submit_request(self, p, req_uuid):
    response_text = self.db.load_data("blackboards", req_uuid, "response")
    if not response_text:
      self.db.save_data("blackboards", req_uuid, "response", json.dumps({'uuid': 'pending'}))
    subprocess.Popen(['python', '-c', f'''from utils.async_ai import Agent; agent = Agent(print_tokens=True); agent.process_request("{req_uuid}", {p})'''])
    return True
