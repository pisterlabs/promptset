import sqlite3
from bs4 import BeautifulSoup
import uuid
import random
import google.generativeai as palm
from openai import OpenAI
import google.generativeai as palm
from .AgentInterface import AgentInterface, AgentTerminalInterface, Node


class TestAgent(AgentInterface):
  def __init__(self, **params):
      super().__init__(**params)
      # Set predefined attributes and any additional attributes from params
      for key, value in params.items():
          setattr(self, key, value)
      # Set defaults for non-provided attributes
      self.thread_id = params.get('thread_id', str(uuid.uuid4())[:8])
      self.input_table = params.get('input_table', 'Messages').split(',')
      self.output_table = params.get('output_table', 'Messages')
      self.test_message = params.get('test_message', f'This is a test message from Agent {self.role}')
      self.rows = params.get('rows', None)
      self.connection = params.get('db_connection', sqlite3.connect(':memory:'))


  def send_message(self, message=None):
      message = message or self.test_message
      connection = sqlite3.connect(self.db_path)
      cursor = connection.cursor()
      cursor.execute(f"INSERT INTO {self.output_table} (thread_id, sender, content) VALUES (?, ?, ?)",
                     (self.thread_id, self.role, message))
      connection.commit()
      connection.close()
      return message

  def execute(self, message=None):
      self.send_message(message)
      return message



# Moderator class
class TestModerator(TestAgent):
  def execute(self):
      if 'turns' not in self.context.keys():
          self.context.turns = {}
      if self.thread_id not in self.context.turns.keys():
          self.context.turns[self.thread_id] = 0

      if self.children:
          # Randomly selecting a child agent to execute next
          next_agent = random.choice(self.children)
          self.send_message(f"{self.role} chose {next_agent.role}.")
          self.context.turns[self.thread_id] += 1
          if self.context.turns[self.thread_id] > 5:
              self.send_message(f"Moderator {self.name} has reached maximum number of turns.")
              self.context.exit = True
          return next_agent.name
      else:
          self.send_message(f"Moderator {self.name} has no participants to choose.")




class OpenAIAgent(AgentInterface):
    def __init__(self, model=None, context=None, **params):        
       
        self.context = context
        super().__init__(**params)
      
    def setup(self):
        # Check if the API key is available in the context
        if hasattr(self.context, 'openai_key') and self.context.openai_key:
            self.client = OpenAI(api_key=self.context.openai_token)
        else:
            raise ValueError("API key not provided in the context. Please set context.openai_key with your API key.")

        
        
    def send_message(self, messages=None):
        messages = messages or self.get_inputs()
        system_prompt = [{'role': 'system', 'content': self.prompt}]
        messages = system_prompt + [{'role': 'user', 'content': msg} for msg in messages]
        openai_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
    
        response = self.parse_response(openai_response.choices[0].message.content)
        self.output_messsage(response)
  
  
        return response

class OpenAIDeveloper(OpenAIAgent, AgentTerminalInterface):
  def parse_response(self, response):
    return AgentTerminalInterface.parse_response(self, response)

  
class PalmAgent(AgentInterface):
  
    def setup(self):
        if hasattr(self.context, 'palm_key') and self.context.palm_key:
            palm.configure(api_key=self.context.palm_key)
        else:
            raise ValueError("API key not provided in the context. Please set context.openai_key with your API key.")
                
    def format_messages(self, messages):
        """
        This method is for formatting messages before being sent to the API.
        the messages arg is a list of strings. 
        """
        if self.prompt: # if there is a prompt load it with the rest of the message thread
          messages = [self.prompt] + messages
        formatted_messages = [{'author': idx%2, 'content': msg} for idx,msg in enumerate(messages)]
        return formatted_messages
    
    def send_message(self, messages=None):
        messages = messages or self.get_inputs()
        messages = self.format_messages(messages)
        response = palm.chat(messages=messages)
        response = self.parse_response(response.messages[-1]['content'])
        return response.messages[-1]['content']


class PalmDeveloper(PalmAgent, AgentTerminalInterface):
  def parse_response(self, response):
    return AgentTerminalInterface.parse_response(self, response)


class SubprocessAgent(AgentInterface):
  def __init__(self, context=None, **params):
      self.client = OpenAI()
      self.context = context
      super().__init__(**params)

  def send_message(self, messages=None):
      messages = messages or self.get_inputs()
      prompt = self.prompt or 'You are a helpful assistant.'
      system_prompt = [{'role': 'system', 'content': prompt}]
      messages = system_prompt + [{'role': 'user', 'content': msg} for msg in messages]

      responses = []

      for message in messages:
          content = message['content']

          if content.strip() == "'''python":
              python_code = self.extract_code_block(messages)
              if python_code:
                  response = self.execute_python_code(python_code)
                  responses.append(response)
          elif content.strip().startswith("<commandline>"):
              command = self.extract_command_from_xml(content)
              if command:
                  response = self.execute_terminal_command(command)
                  responses.append(response)

      self.insert_responses_into_db(responses)
      return responses

  def extract_code_block(self, messages):
      code_block = []
      is_inside_code_block = False

      for message in messages:
          content = message['content']

          if is_inside_code_block:
              if content.strip() == "'''":
                  return '\n'.join(code_block)
              else:
                  code_block.append(content)
          elif content.strip() == "'''python":
              is_inside_code_block = True

      return None

  def execute_python_code(self, code):
      try:
          result = subprocess.check_output(['python', '-c', code], stderr=subprocess.STDOUT, text=True)
          return result
      except subprocess.CalledProcessError as e:
          return f"Python Error: {e.output}"

  def extract_command_from_xml(self, xml_content):
      try:
          root = ET.fromstring(xml_content)
          if root.tag == "commandline":
              return root.text.strip()
          else:
              return None
      except ET.ParseError:
          return None

  def execute_terminal_command(self, command):
      try:
          result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
          return result
      except subprocess.CalledProcessError as e:
          return f"Command Error: {e.output}"

  def insert_responses_into_db(self, responses):
      cursor = self.connection.cursor()
      for response in responses:
          cursor.execute(f"INSERT INTO {self.output_table} (thread_id, sender_id, content) VALUES (?, ?, ?)",
                         (self.thread_id, self.role, response))
          self.connection.commit()
