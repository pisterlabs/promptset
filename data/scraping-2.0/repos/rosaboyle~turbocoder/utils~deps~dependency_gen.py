import os
import json
import openai
# openai.api_key = ""
prompt = """Here is the code. I don't know what language it is written in. I want you to find the language it is written in. Then get the dependency file for the code. The output should be in the format of json with keys "lang", "depsFileName","fileContent"

{}

I just want you to give me the json and nothing else. Don't explain me. Don't tell me anything else.
"""


prompt_docker = """Now given the dependency file and the code. I want you to generate a Dockerfile for the code. 

{}

 I just want you to give me the output of the file and nothing else. Don't explain me. Don't tell me anything else.
"""

prompt_environment_vars = """Now given the dependency file and the code. I want you to generate environment variables required to succesfully run the code. 

{}

 I just want you to give me the output Environment variables and nothing else. Don't explain me. Don't tell me anything else.
"""


prompt_command_gen = """Now given the dependency file and the code. I want you to generate a Command required to successfully run the code. 

{}

 I just want you to give me command and nothing else. Don't explain me. Don't tell me anything else.
"""



def get_prompt(code):
    return prompt.format(code)
    
def get_prompt_dockerfile(code):
    return prompt_docker.format(code)

def get_prompt_command_gen(code):
    return prompt_docker.format(code)
 
def is_valid_json(json_str):
    required_keys = ["lang", "depsFileName", "fileContent"]
    
    try:
        # Parse the JSON string
        data = json.loads(json_str)
        
        # Check if all required keys are present
        if all(key in data for key in required_keys):
            return True
        else:
            return False
    except json.JSONDecodeError:
        return False
      
def get_deps(code):
  prompt_in = prompt.format(code)
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_in}
  ]
  )
  return completion.choices[0].message.content

def get_dockerfile(code):
  prompt_in = get_prompt_dockerfile.format(code)
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_in}
  ]
  )
  return completion.choices[0].message.content

def get_command(code):
  prompt_in = get_prompt_command_gen.format(code)
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_in}
  ]
  )
  return completion.choices[0].message.content


def get_environment_vars(code):
  prompt_in = prompt_environment_vars.format(code)
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_in}
  ]
  )
  return completion.choices[0].message.content



def json2file(output_json):
    file_name = output_json["depsFileName"]
    file_content = output_json["fileContent"]
    # Writing to the file
    with open(file_name, 'w') as file:
        file.write(file_content)
