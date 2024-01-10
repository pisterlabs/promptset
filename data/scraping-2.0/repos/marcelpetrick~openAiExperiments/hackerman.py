import os
import openai
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------

def gpt4request():
  #gpt_prompt = input ("What prompt do you want to use: ")
  #print(f"Working with gpt_prompt: {gpt_prompt}")
  gpt_prompt = "Write me a python program which prints the very first 100 prime numbers. Not up to 100, but the first 100 ones."
  start_time = time.time()

  message=[{"role": "user", "content": gpt_prompt}]
  response = openai.ChatCompletion.create(
      model="gpt-4-0613",
      messages = message,
      temperature=0.2,
      max_tokens=1000,
      frequency_penalty=0.0
  )
  end_time = time.time()

  elapsed_time = end_time - start_time
  print(f"The code took {elapsed_time} seconds to run.")
  print(f"response: {response}")

if __name__ == "__main__":
  for a in range(0,9):
    gpt4request()

# example prompt:
# Write me a python program which prints the very first 100 prime numbers. Not up to 100, but the first 100 ones.

#
# gpt-4: uses gpt-4-0314 -> 26s
# gpt-4-0613 -> 16s
# gpt-3.5-turbo-16k-0613 -> 27s
