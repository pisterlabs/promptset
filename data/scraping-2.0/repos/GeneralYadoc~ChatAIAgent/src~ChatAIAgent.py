import queue
import time
import math
from typing import Callable
from dataclasses import dataclass
import threading
import openai

@dataclass
class userMessage:
  message: str = ""
  extern: any = None

@dataclass
class params:
  api_key: str
  system_role: str = "You are a helpful assistant."
  ask_cb: Callable[[userMessage,], None] = None
  max_messages_in_context: int = 20
  answer_cb: Callable[[userMessage, any], None] = None
  max_queue_size: int = 10
  model: str = 'gpt-3.5-turbo'
  max_tokens_per_request: int = 256
  interval_sec: float = 20.0

class ChatAIAgent(threading.Thread):
  def __init__( self, params ):
    self.__user_message_queue = queue.Queue(maxsize=params.max_queue_size)
    openai.api_key = params.api_key
    self.__current_context = [{"role": "system", "content": params.system_role}]
    self.__ask_cb = params.ask_cb
    self.__max_messages_in_context = params.max_messages_in_context
    self.__interval_sec=params.interval_sec
    self.__answer_cb = params.answer_cb
    self.__model = params.model
    self.__max_tokens_per_request = params.max_tokens_per_request
    self.__keeping_connection = False
    super(ChatAIAgent, self).__init__(daemon=True)

  def connect(self):
    self.start()
    self.join()

  def disconnect(self):
    self.__keeping_connection = False

  def put_message(self, user_message):
    if self.__user_message_queue.full():
      self.__user_message_queue.get()
    self.__user_message_queue.put(user_message)

  def full_messages(self):
    return self.__user_message_queue.full()

  def run(self):
    self.__keeping_connection = True
    while self.__keeping_connection:
      start_time = time.time()
      while self.__keeping_connection and not self.__user_message_queue.empty():
        user_message = self.__user_message_queue.get()
        if self.__ask_cb:
          self.__ask_cb(user_message)
        self.__current_context.append({"role": "user", "content": user_message.message})
        if len(self.__current_context) >= self.__max_messages_in_context:
          self.__current_context.pop(1)
        completion = None
        while self.__keeping_connection and not completion:
          try:
            completion = openai.ChatCompletion.create(model=self.__model, max_tokens=self.__max_tokens_per_request, messages=self.__current_context)
          except:
            completion = None
          time.sleep(0.1)
        if completion:
          if self.__answer_cb:
            self.__answer_cb(user_message=user_message, completion=completion)
          self.__current_context.append({"role": "assistant", "content": completion.choices[0]["message"]["content"]})
          if len(self.__current_context) >= self.__max_messages_in_context:
            self.__current_context.pop(1)
        self.__sleep_from(start_time)
        start_time = time.time()
      self.__sleep_from(start_time)
      start_time = time.time()

  def __sleep_from(self, start_time, interval_sec=None):
    if not interval_sec:
      interval_sec = self.__interval_sec
    cur_time = time.time()
    sleep = interval_sec - (cur_time - start_time)
    if sleep > 0.:
      sleep_counter = math.floor(sleep * 10)
      sleep_frac = sleep - (sleep_counter / 10.)
      for i in range(sleep_counter):
        if not self.__keeping_connection:
          break
        time.sleep(0.1)
      time.sleep(sleep_frac)