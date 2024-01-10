import json
import openai
from os import path

current_directory = path.dirname(path.abspath(__file__))
secrets_path = path.join(current_directory, '..', 'secrets.json')

try:
    with open(secrets_path, 'r') as f:
        secrets = json.load(f)
    # secrets = json.loads(open(path.join(current_directory, '..', 'secrets.json'), "r"))
except Exception as e:
    print(e)
    print('''Please make a secrets.txt with an OpenAI API key in it. Format: `{
  "OPENAI_API_KEY": "key-here"}`''')

openai.api_key = secrets.get("OPENAI_API_KEY")

def get_tools(prompt, tools):
    print('------')
    print('PROMPT', prompt)
    print('TOOLS', tools)
    content_tmpl = '''I have a prompt from a user, and a list of tools that they could use. Given their prompt, please list the names of the tools that best fit. Give the list of tools in a json array of tool names.
            
            Prompt: {prompt}
            
            Tools: {tools}

    Do not response except for a json array of tools, formatted like: ["tool1", "tool2"]. If you do not have any tools to suggest, please return an empty array.
            '''
    # content.format(prompt=prompt, tools=tools)
    content = content_tmpl.format(prompt=prompt, tools=tools)
    print('content', content)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",  "content": content},
        ]
    )
    # check for a successful response

    return response['choices'][0]['message']['content']
    