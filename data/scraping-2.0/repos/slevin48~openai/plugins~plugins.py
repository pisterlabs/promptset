import openai
import json
import requests
import os


def chat(prompt,system = '', plugin=[]):
  # Generate a response from the ChatGPT model
  messages = []
  if plugin:
    plugins_list = load_plugins()
    # plugins_list = load_plugins(plugin) # TODO: insert plugin list in the function call
    plugin_name = pick_plugin(prompt,plugins=plugins_list)
    if plugin_name == 'search':
      system = search_plugin(prompt)
      print(f"Plugin: search \n System context: {system}")
    elif plugin_name == 'MATLAB':
      system = matlab_plugin()
      print(f"Plugin: MATLAB \n System context: {system}")
    else: # Probably "null"
      print('sorry, could not use a plugin for this request')
  if system:
    messages.append({'role': 'system', 'content': system})
  messages.append({'role': 'user', 'content': prompt })
  completion = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
        messages= messages
  )
  return completion.choices[0].message.content

def load_plugins():
  # TODO: update calling sequence to take a list of plugins
  # load_plugins(['search','MATLAB'])

  plugins_list = [
            {'name':'search',
            'desc' : 'Useful for when you need to answer questions about current events.',
            'example' : 'What movie won best picture in 2023?'
            }
           ,{'name':'MATLAB',
            'desc' : 'Useful for when performing numerical computing and trying to solve mathematical problems',
            'example' : 'What is the 10th element of the Fibonacci suite'
            }
           ]

  return plugins_list

def pick_plugin(prompt,plugins):
  # Use ChatGPT to decide which plugin to use
  # Arg:
  # prompt : str 
  # plugins : List of Dict for each plugin
  # Out: 
  # plugin_name : str of the plugin name

  name = [p['name'] for p in plugins]
  desc = {p['name']:p['desc'] for p in plugins}
  examples = {p['name']:p['example'] for p in plugins}

  promptIntro = '''You are an AI language model that decides whether extra information is needed before you would be able to respond to a hypothetical prompt. 
  For every prompt you receive, you will decide whether one of the available plugins to use.
  '''
  promptCore = ''.join([f'- {n} : {desc[n]}\n' for n in name])
  promptExample = '''
  Here's some examples of prompts you'll get and the response you should give:

  ''' + ''.join([f'  USER: {examples[n]}\n  BOT: {n}\n\n' for n in name])
  promptEnd = '''
  Give a single word answer with the tool chosen to answer the prompt.
  If you're very confident that you don't need extra information, respond with the string "null".
  '''

  pluginDecisionPrompt = promptIntro+promptCore+promptExample+promptEnd 
  plugin_name = chat(prompt,system=pluginDecisionPrompt)
  return plugin_name

def search_plugin(prompt):
  # Set the API endpoint
  api_endpoint = "https://serpapi.com/search"
  # Set your API key
  api_key = os.environ['SERP_API_KEY']
  # Set the search parameters
  params = {
      "q": prompt,
      "api_key": api_key,
  }
  # Send the request
  response = requests.get(api_endpoint, params=params)
  results = json.loads(response.content)

  # Build system promt from featured snippet answer

  searchPrompt = results["answer_box"]["answer"]

  system = f'''Answer the user request given the following information retrieved from an internet search:
  {searchPrompt}
  '''
  return system

def matlab_plugin():
  system = '''Generate MATLAB code from the request.
  Give only the code, no explanation in natural language.
  '''
  return system