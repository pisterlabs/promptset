import replicate
import os
from openai import OpenAI
import argparse
import random

job = random.randint(1, 100000)

oaiclient = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


DEFAULT_FILTER = "you are a video summerizer and you will recieve a video transcription from whisper to summerize"
def get_response(prompt):
  """
  Retrieves a response by splitting the prompt into chunks of 50k characters,
  sending each chunk to an API for completion, and combining the responses
  into a single string.

  Args:
    prompt (str): The prompt to be sent for completion.

  Returns:
    str: The combined response from the API.

  Raises:
    None
  """
  MAX_CHARS = 50000  # Maximum characters per API call
  responses = []

  # Split the prompt into chunks of 50k characters
  prompts = [prompt[i:i+MAX_CHARS] for i in range(0, len(prompt), MAX_CHARS)]

  for prompt in prompts:
    prompt = "Please summerize the following video which has been transcribed by whisper. Use and tell me important quotes so it feels as if i watched it: \n" + prompt
    messages = [{"role": "system", "content": DEFAULT_FILTER}] + [{"role": "user", "content": prompt}]
    completion = oaiclient.chat.completions.create(model="gpt-4-1106-preview", messages=messages)
    message = completion.choices[0].message
    responses.append(message.content)

  # Combine the responses into a single string
  response = ' '.join(responses)
  return response
parser = argparse.ArgumentParser(description='make a description of a video')
parser.add_argument('--url', type=str, required=True,
                        help='youtube url')
args = parser.parse_args()
url = args.url

# Run audio through the API
output = replicate.run(
  "adidoes/whisperx-video-transcribe:481284a2a2ff72a031689481ca92fb1d20b194980a4b435d93f6f4c9520fea61",
  input={
    "url": url,
    "debug": False,
    "batch_size": 16
  }
)
# save output to a text file
with open(f'transcript{job}.txt', 'w') as f:
    f.write(output)
# open file and read it and split it into 50k character chunks and then process them through gpt4-turbo to get a summmery and then if there are multiple summeries combine them into one and print the summery
with open(f'transcript{job}.txt', 'r') as f:
    text = f.read()
    print(get_response(text))
os.remove(f'transcript{job}.txt')
