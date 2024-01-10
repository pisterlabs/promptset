
import os
import time
import openai

# our default gpt chat call
def chat_completion(system_prompt, user_prompt):
  openai.api_key = os.getenv("OPENAI_API_KEY")

  start = time.time()
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": user_prompt,
      }
    ],
    temperature=1.0,
    stream=True
  )

  output = ''
  for event in response:
    content = event["choices"][0].get("delta", {}).get("content")
    if content is not None:
      output += content
      print(content, end='')
  print(f"\nDone in {(time.time() - start):.2f}")

  return output
