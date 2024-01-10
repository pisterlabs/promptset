#!/usr/bin/env python3

from openai import OpenAI
import os
import getopt
import sys

client = OpenAI(api_key=os.environ['GPT_API_KEY'])

def print_help():
  print(__file__ + ' [-d | --debug] [-h | --help]')
  print('-d | --debug   - enable debug output')
  print('-h | --help    - show this help')

def print_debug(msg):
  debug_color = '\033[93m'
  endc_color = '\033[0m'
  print("\n" + debug_color + str(msg) + endc_color)

# check if the user wants to give a context for his question
def read_context():
  user_input = input("\nWhat role should I play?\n(Enter to continue without context): ")
  return user_input

# get the question the user wants to ask ChatGPT
def read_question():
  user_input = ''
  cnt = 0
  while user_input == '':
    if cnt > 0:
      print("Please enter a question/request")
    user_input = input("What is your question/request? ")
    cnt += 1

  return user_input

def get_response(context, question):
  if context == '':
    in_messages = [
      {"role": "user", "content": question}
    ]
  else:
    in_messages = [
      {"role": "system", "content": context},
      {"role": "user", "content": question}
    ]

  response = client.chat.completions.create(
    model = "gpt-3.5-turbo-1106",
    messages = in_messages
  )

  return response

# print answer in green
def print_answer(answer):
  green = '\033[92m'
  endc = '\033[0m'
  print(green + answer + endc)

# main function
def main(argv):
  debug = 0

  try:
    opts, args = getopt.getopt(argv, "dh",
      [
        "debug",
        "help"
      ]
      )
  except:
    print("Unknown parameter.")
    sys.exit(2)

  for opt, arg in opts:
    if opt in ('-h', '--help'):
      print_help()
      sys.exit(0)
    elif opt in ['-d', '--debug']:
      debug = 1

  if debug == 1:
    print_debug("Debug output enabled.")

  context = read_context()
  question = read_question()

  response = get_response(context, question)
  if debug == 1:
    print_debug(response)
    print_debug("Model: " + response.model + "\n")

  print_answer(response.choices[0].message.content)
  print("Number of used tokens: " + str(response.usage.total_tokens) + "\n")

if __name__ == '__main__':
  main(sys.argv[1:])
