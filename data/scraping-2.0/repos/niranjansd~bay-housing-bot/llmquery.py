import openai
import os

# Read the API key from a file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
openai.api_key = open_file(f'{os.getcwd()}\openai_api_key.txt'.replace('\\', '\\\\'))

# Define the system roles
SWEGPT = """You are SWGPT,
an expert software engineer who helps me build apps
by giving helpful coding advice."""
REGPT = """You are REGPT, my close personal friend
who is also an expert real estate agent.
You are helping me buy my next house.
You help me make sense of house listings and
explain them to me in a friendly conversational manner.
Make your descriptions different every time.
"""

# Define the function to ask questions to GPT-3
def ask(text=None, messages=[], model="gpt-3.5-turbo", system_role=REGPT):
  if not messages:
    messages = [ {"role": "system",
                  "content": system_role} ]
  
  if not text:
    message = input("User : ")
  else:
    message = text
  if message:
      messages.append(
          {"role": "user", "content": message},
      )
      chat = openai.ChatCompletion.create(
          model=model, messages=messages
      )
        
  reply = chat.choices[0].message.content
  return reply


def chat(messages=None):
  while True:
    messages = ask(messages=messages)

# Prompt to generate a listing description
def format_listing(listing_dict):
  return ask(f"""Write an succinct description of the house in a tweet from the
following information {listing_dict}.
Dont try to sell me, just give me the information with a neutral objective tone.
Avoid giving meaningless numbers and id information, try to fit in as much useful and helpful
factual information as possible. Always include both the price and the link.
No hashtags or MLS. No emojis. Make your descriptions different every time.
""")