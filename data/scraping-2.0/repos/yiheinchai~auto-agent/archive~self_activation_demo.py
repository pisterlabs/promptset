import openai
import time


class LLMRecursiveCaller:

  def __init__(self, api_key, prompt, engine="text-davinci-003"):
    print('initialising...')
    self.api_key = api_key
    self.prompt = prompt
    self.engine = engine

  def generate_output(self, prompt):
    print('calling LLM with prompt: ', prompt)
    response = openai.Completion.create(prompt=prompt,
                                        api_key=self.api_key,
                                        engine=self.engine,
                                        max_tokens=1024,
                                        n=1,
                                        stop=None,
                                        temperature=0.5)
    output = response.choices[0].text.strip()
    print("=========")
    return output

  def call_recursive(self):
    output = self.generate_output(
      self.prompt) + "PROMPT: Make sure to not continue any lists. Give instruction on what should be done next: "
    new_call = LLMRecursiveCaller(self.api_key, output, self.engine)
    time.sleep(3)
    if output:
      new_call.call_recursive()
    else:
      print("no output")
      return ""


api_key = "<YOUR API KEY>"
prompt = "Talk as if you are chatting with someone"
recursive_caller = LLMRecursiveCaller(api_key, prompt)
output = recursive_caller.call_recursive()
print("Completed")
