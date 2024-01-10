#!/usr/bin/env python3.8
import os
import openai
import dotenv

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class convo:
  def __init__(self):
    self.context = ['']*6

    
  def query(self,question):
    question = ''.join(self.context)+question
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=question,
      temperature=0.6,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)
    self.add_context(response.choices[0].text,question)
    return "Mathilda:"+response.choices[0].text
  
  def add_context(self,mathilda, user):
    self.context[:4] = self.context[2:]
    self.context[4] = user
    self.context[5] = mathilda

def test_prompt():
  response = openai.Completion.create(
  model="text-davinci-002",
  prompt="Say hi I'm mathilda",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0)
  return response.choices[0].text

