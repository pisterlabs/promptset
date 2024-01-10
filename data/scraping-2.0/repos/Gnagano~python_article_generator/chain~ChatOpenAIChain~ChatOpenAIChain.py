import os
from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

class ChatOpenAIChain (ABC):
  TEMPLATE_NAME = ''
  
  def __init__(self, model="gpt-3.5-turbo"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir_path = os.path.join(current_dir, '../template')
    template_file_path = os.path.join(template_dir_path, f"{self.TEMPLATE_NAME}.txt")
    with open(template_file_path, 'r') as f:
      prompt = PromptTemplate.from_template(f.read())
      chat = ChatOpenAI(model=model)
      self.chain = LLMChain(llm=chat, prompt=prompt)

  @abstractmethod
  def get_response(self, **kwargs):
      pass