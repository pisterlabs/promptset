import os
import json
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")

class Role:
  def __init__(self, messages, functions):
    self.client = OpenAI(api_key=api_key)
    self.messages = messages
    self.functions = functions

  # GPT-3.5와 상호작용하는 함수
  def interact(self, message):
    self.messages.append({ "role": "user", "content": message })
    
    response = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=self.messages,
      functions=self.functions,
      function_call="auto"
    )

    self.messages.append(response.choices[0].message)

    if response.choices[0].message.function_call is not None:
      return {
        "function": response.choices[0].message.function_call.name,
        "args": json.loads(response.choices[0].message.function_call.arguments)
      }
    else:
      return response.choices[0].message.content
