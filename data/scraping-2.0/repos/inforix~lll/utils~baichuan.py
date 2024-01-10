import torch
from transformers import Pipeline, pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from typing import List, Optional, Mapping, Any
import sys
import logging

class BaichuanChat(LLM):
  max_token: int = 10000
  temperature: float = 0.1
  top_p = .9
  messages = []
  tokenizer: object = None
  model: object = None
  model_path:str = None
  device: str = None
  
  def __init__(self, model_name_or_path:str, device:str = None):
    super().__init__()
    self.model_path = model_name_or_path
    self.device = "cuda" if torch.cuda.is_available() else "cpu" if device is None or device == "cuda" else device
    
    self.load_model()
    
  @property
  def _llm_type(self) -> str:
    return "ChatGLM"
  
  def _call(self, prompt:str, stop: Optional[List[str]] = None) -> str:
    self.messages.append({"role": "user", "content": prompt})
    response = self.model.chat(
      self.tokenizer,
      self.messages
    )
    
    if stop is not None:
      response = enforce_stop_tokens(response, stop)
    print(response)
    
    self.messages.append({"role": "bot", "content": response})
    
    return response

  def load_model(self):
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

    if self.device == "cuda":
      self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16).cuda()
    else:
      self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).float()
    
    self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
    self.model = self.model.eval()

if __name__ == "__main__":
  torch.cuda.is_available = lambda: False
   
  llm = BaichuanChat("../models/Baichuan-13B-Chat")

  answer = llm.generate(["你好"])
  print(answer)
  