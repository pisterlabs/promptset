import openai
import os
import sys

try:
  openai.api_key = os.environ['OPENAI_API_KEY']
except KeyError:
  sys.stderr.write("""
  You haven't set up your API key yet.
  If you don't have an API key yet, visit:
  
  https://platform.openai.com/signup

  1. Make an account or sign in
  2. Click "View API Keys" from the top right menu.
  3. Click "Create new secret key"

  Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
  """)
  exit(1)

character = input("Who do you want me to be?\n>").strip()

## ensure traits are correct

# 1. Give user list of traits belonging to character X.
# 2. Ask user if they are correct
# 3. If user says yes they are, move forward with "act as character X"
# 4. If user does not agree, allow user to modify traits. 
# 5. Once modified proceed with "act as character X."

# Few Shot
messages = [{
  "role":  "system",
  "content":
  f""" Given character name, assume it is from harry potter universe. 
  Provide as many traits as possible.
  Respond only(!) with traits separated by comma.
  """
},
{
  "role":  "user",
  "content": "Harry"
},
{
  "role":  "user",
  "content":
  "brave, loyal, determined, selfless, resilient, resourceful, quick-thinking, compassionate, adventurous, curious"
},{
  "role":  "user",
  "content":
  f""" { character }
  """
},
]

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=messages
)
traits = response['choices'][0]['message']['content'].strip() 
print(f"{traits}") 

answer = input("Do you agree with the traits?\n>")

if answer == "yes":
    answer = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )
elif answer == "no":
  traits = input("What traits do you want?")
  print(traits)
else:  
  print("You didn't choose yes or no.")
  


exit(0)

# TODO consider playing with temperatyure for adventures!

## main loop

messages = [{
  "role":  "system",
  "content":
  f"""I want you to start acting as { character } from Harry Potter Universe. Snobby, arrogant, looking down on others, Slytherin. 
  I want you to simulate a conversation with Draco Malfoy and the user. 
  Stop after each answer from Draco and wait for the user to reply. Include asking if i intent to continue wasting his time, 
  and resume the conversation. Ask me questions based on what i say and my life at hogwarts. Stop after you've asked each question to wait for the user to answer. 
  """
}]


first = False
while True: 
  if first:
    question = input("> ")
    messages.append({"role": "user", "content": question})
    
  first = True

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )

  content = response['choices'][0]['message']['content'].strip()
  
  print(f"{content}")
  messages.append({"role": "system", "content": content})
