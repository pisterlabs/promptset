import os 
import openai 
import argparse 

TEMPERATURE=0.89
MAX_TOKENS=162
TOP_P=1
FREQUENCY_PENALTY=0
PRESENCE_PENALTY=0.6

class ChatBot(object):
  def __init__(self, api_key, model_name):
    openai.api_key = api_key
    self.history = "JOY-> Hello, I am your personal mental health assistant. What's on your mind today?\n"
    print(self.history)
    self.model_name = model_name

  def start_conversation(self):
    user_input = input("User-> ")
    while user_input not in ['bye','exit']:
      self.history += f"User-> {user_input}JOY->"
      response = openai.Completion.create(model=self.model_name,
                                          prompt=self.history,
                                          temperature=TEMPERATURE,
                                          max_tokens=MAX_TOKENS,
                                          top_p=TOP_P,
                                          frequency_penalty=FREQUENCY_PENALTY,
                                          presence_penalty=PRESENCE_PENALTY,
                                          stop=["JOY","User","USER","Joy"])
      response = response['choices'][0].get('text')
      response = response.replace('END','').replace('END_TO_END','')
      answer = f"JOY-> {response}"
      self.history += f"{response}\n"
      print(answer)
      user_input = input("User-> ")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--api-key', type=str, help='API key generated from openai')
  parser.add_argument('--model-name', type=str, help='Model name received after training')
  args = parser.parse_args()
  consultant = ChatBot(api_key = args.api_key,
                      model_name=args.model_name)
  consultant.start_conversation()
