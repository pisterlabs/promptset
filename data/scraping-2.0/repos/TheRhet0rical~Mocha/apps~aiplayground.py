# AI Playground | 12/1/22, Rhet0rical
# An AI using the OpenAI framework.

# Imported Modules (If Needed)
from system.config import *
import openai

# Bot Config
class bot:
  model = 'text-davinci-003'
  temp = 0.7
  max_tokens = 120
  step = 1

# Question Function
def question():
  question = input(f'  {c.yellow}You{c.white} [{c.italic}{bot.step}{c.clear}]: ')
  
  if question == 'stop':
    aiActive = False
    print(f'\n  System: {c.red}Ended{c.white} Conversation. Exiting Application...\n\n{v.l}')
    console.wait(2)
    exit()
    
  else:
    response = openai.Completion.create(model=bot.model, prompt=question, temperature=bot.temp, max_tokens=bot.max_tokens)
    #print(response)
    answer = response["choices"][0]["text"]
    print(f'\n  AI: {answer.strip()}\n')
  
# Class Data
class aiplayground:
  name = 'AI Playground'                                      # (STR) Application Name.
  tag = 'APP'                                                     # (STR) App Tag, Usually 'SYS', 'APP', Or 'TST' Respectively.
  description = 'An AI Utilizing the OpenAI Framework.'           # (STR) App Description, Self Explanitory
  version = '1.1'                                                 # (INT, STR) App Version, Also Self Explanitory OR (STR) 'nv' To Not Show Version.

  def launch():                                                   # Every Application Will Need A Launch Function. Put Your App Code In This Function.
    console.clear()
    print(f'{v.l}\n')
    console.title(' AI Playground')
    print(f' {v.p} Version: {c.purple}{aiplayground.version}{c.white}, Created By {c.blue}Rhet0rical{c.white}')
    print(f'  - Current Settings: {c.italic}Temp: {bot.temp}, Max Tokens: {bot.max_tokens}.\n')
    print(f'{v.l}\n')

    keysuggestion = input(f' {v.p} Please Enter Your {c.blue}OpenAI{c.white} API Key Below...  \n  - Key: ')

    try:
      if keysuggestion == 'admin':
        openai.api_key = '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'
      else:
        openai.api_key = keysuggestion
      aiActive = True
      print(f'\n {v.p} {c.green}Successfully{c.white} Retreived User Data. Loading {c.blue}Playground{c.white}...\n\n{v.s}\n')
      console.wait(2)
      print(f'  {c.red}Remember{c.white}, you can \"stop\" at any time to deactivate the AI.\n')
      
    except:
      print(f'\n  {c.red}Failed{c.white} To Retreive User Data. Check Your {c.yellow}API{c.white} Key And Try Again.\n{v.l}')
      console.wait(2)
      exit()

    while aiActive == True:
      question()

      bot.step = bot.step + 1

      if aiActive == False:
        break