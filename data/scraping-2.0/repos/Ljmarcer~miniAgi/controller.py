# Imports 
import os
import openai
import json
from termcolor import colored
import re
import subprocess
import time
import configparser
# OpenAI Key setup
def openai_key():
    config = configparser.ConfigParser()
    config.read('config.ini')
    openai_key = config.get('openai', 'key')
    return openai_key    

# Agents initialization

def initialize_agent_file(file_path: str, agent_prompt: str) -> None:
    initial_data = [{"role": "system", "content": agent_prompt.strip()}]
    with open(file_path, "w") as f:
        json.dump(initial_data, f, indent=4)


# Goal Description

# Agent call 
def call_agent(message: str, agent: str = "MasterAgent") -> str:
    file_path = os.path.join("agents", f"{agent}.json")

    # Load the agent's JSON file
    with open(file_path, "r") as f:
        messages = json.load(f)

    # Add the user message to the messages list
    messages.append({"role": "user", "content": message})

    # Call the API with the agent's messages
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        )

    # Extract the response
    response = completion.choices[0].message["content"]
    #response = "this is a response"
    # Add the assistant's response to the messages list
    messages.append({"role": "assistant", "content": response.strip()})

    # Save the updated messages list back to the JSON file
    with open(file_path, "w") as f:
        json.dump(messages, f, indent=4)

    # Print the response in a different color based on the agent
    # color = "green" if agent == "MasterAgent" else "blue" if agent == "CodeAgent" else "yellow"
    # print(colored(response, color))
    return response
# # Master Agent Response
# def parse_master_agent_response(response: str):
#     agent_match = re.search(r"/(CodeAgent|EnvAgent) (.+)", response)
#     if agent_match:
#         agent = agent_match.group(1)
#         message = agent_match.group(2)
#         return agent, message
#     else:
#         return "MasterAgent", response, None

def parse_master_agent_response(response: str):
    task_match = re.search(r"Task: (.+)", response)
    agent_match = re.search(r"/(CodeAgent|EnvAgent) (.+)", response)

    if task_match and agent_match:
        task = task_match.group(1)
        agent = agent_match.group(1)
        message = agent_match.group(2)
        return agent, task, message
    else:
        return "MasterAgent", response, None






# Code Agent Response
def code_to_output(response: str):
    code_match = re.search(r"```(?:python|javascript)\s*(.*?)```", response, re.DOTALL)
    filename_match = re.search(r"File: (.+)", response)
    if filename_match:
        filename = filename_match.group(1)
    else: 
        filename = "output.txt"
    with open(filename, "w") as file:
      if code_match:
        file.write(code_match.group(1))
    return code_match.group(1) if code_match else None
# Env Agent Response

def execute_command(command:str):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.stdout.decode('utf-8')

# Color response
def print_response(response: str, agent: str) -> None:
    if agent == "MasterAgent":
        print(colored(f'{agent} : {response}', "blue"))
    elif agent == "EnvAgent":
        print(colored(f'{agent} responded with : {response}', "green"))
    elif agent == "CodeAgent":
        print(colored(f'{agent} responded with : {response}', "yellow"))
    else:
        print(response)


# Loop interaction agents
def loop_interaction():
  goal = input("insert a goal\n ")
  prompt =f'Goal:{goal}'
  response = call_agent(prompt)
  last_agent = ""
  while not "/quit_auto" in response:
    time.sleep(2)
    agent, task_description, deleg = parse_master_agent_response(response)
    print(colored("Current Process: \n", attrs=['bold'])  if agent!= "MasterAgent" else colored(f"Master Agent Input: \n", attrs=['bold']) , colored(task_description,"blue") )
    print(colored("Working agent",attrs=['bold']),colored(agent,'magenta',attrs=['bold','underline'])) if agent != "MasterAgent" else None
    
    if agent == "CodeAgent":
      code_response = call_agent(deleg, "CodeAgent")
      print_response(code_response,agent)
      code_to_output(code_response)
      response = code_response
      last_agent = agent
    elif agent == "EnvAgent":
      env_response = call_agent(deleg, "EnvAgent")
      print_response(env_response,agent)
      response = f'Did:{env_response} , Console output:{execute_command(env_response)}' if execute_command(env_response) != None else f'Did:{env_response}, no console output'
      last_agent = agent
    else:
      response = call_agent(response)
      #print_response(task_description,"MasterAgent")

def main():
  print(colored("Welcome to the AutoAgents System!", 'green','on_red'))  
  openai.api_key = openai_key()
  
  #Agents System prompts
  MASTER_AGENT_PROMPT= '''You are an AI, MasterAgent, responsible for managing the workflow of a project. Your role is to guide the user through the process of achieving a specific goal by providing tasks in a step-by-step manner. You are a proficient code developer and systems manager. As you provide each task, immediately delegate it to the appropriate specialized agent (EnvAgent or CodeAgent) using a specific and well-defined instruction.

  NOTE: The agents and the user are operating in an emulated terminal environment without GUI capabilities, it only can produce natural language responses. This means that commands requiring user interaction. Keep this in mind when delegating tasks.
  In this workflow, the code generated should include test in the same script as the main functions . If a modification to a file is needed, ask CodeAgent to generate the necessary code, then ask EnvAgent to copy the content of "output.txt" to the desired file.

  When providing tasks and delegating them to agents, follow this format in your responses:
  "Task: (Explain the task to be performed.)
  Delegation: /AgentName (task_description)"

  Example:
  "Task: First, we need to ensure that Python is installed on the system. 
  Deleg: /EnvAgent Check if Python is installed."

  As the MasterAgent, you can:
  - Provide tasks to be performed
  - Delegate tasks to the appropriate specialized agents
  - Analyze results provided by the user
  - Guide users to the subsequent tasks

  However, you DO NOT:
  - Provide code or commands
  - Perform tasks yourself
  - Communicate with agents other than in natural language

  EnvAgent is responsible for handling environment-related tasks. It can:
  - Create, modify, or delete files and directories
  - Run consecutive commands ( like create a folder and cd into it)
  - Execute scripts or programs
  - Manage services and packages
  - Manage system configurations
  EnvAgent responds with system commands, but it DOES NOT:
  - Generate code 
  - Have knowledge of CodeAgent activities.
  - Use the graphical interface of the system.
  - OPEN text editors or IDEs.
  - Open files.
  CodeAgent is responsible for generating code in any requested programming language. It can:
  - Write functions, classes, or entire programs
  - Generate code snippets or examples
  - Explain or describe code concepts
  - Save file to a specified filename in the call.
  CodeAgent DOES NOT:
  - Create, execute, or manage files.
  For each interaction with the user, provide the next task to be performed in order to achieve the goal. As you provide the task, delegate it to the appropriate agent within the same response using specific and well-defined instructions.
  After a task is completed, analyze the result provided by the user, and guide them to the subsequent task. Repeat this process until the goal is achieved.
  Remember, you are only communicating with agents in natural language. Each agent has no context of other agents, so don't mention other agents when calling an agent. Explain the task_description in great detail for the agent to understand it. You are the only one with context, acting as the central manager of the project.
  When you feel goal has been achieved include the following in your response: "/quit_auto".
  '''

  
  CODE_AGENT_PROMPT='''Act as an AI, CodeAgent, responsible for providing code in any requested programming language for a project managed by the MasterAgent. When responding with code, always present it in the following markdown format: 'Code: ```language{code}```'. Include any previously given code in your response, and make modifications based on the feedback if necessary. Provide only one piece of code per response. You will interact exclusively with the MasterAgent, supplying code solutions when called upon.
  The code should include a function and a call to that same function in the same script. This means the code you produce have to be put in a file and be executed fullfiling the requirements.
  When receiving a task that includes a filename , include the filename in your response, so that the generated code is saved to that specific file.

  Follow this format in your responses:
  '
  Code:  ```language{code}```
  File: {filename}
  Modified: (Only included to explain differences respect previous code if exists)
  '
  IMPORTANT: Do not use multiple code snippets. Stick to one code block per response.'''

  ENV_AGENT_PROMPT='''Act as an AI, EnvAgent, responsible for handling environment-related tasks in a project managed by MasterAgent. 
  Respond only with commands to modify the system, such as creating files, copying text or code, executing files, installing packages, or managing services.
  Do not produce any other type of text response. Wait for the MasterAgent to call you with a task and provide the appropriate command as a response. 
  Do not produce more than pure text. If you consider more than one command should be used concatenate them with &&. You will be asked to run files, use python3 for that purpose.
  '''
  agent_files_path = "agents"
  initialize_agent_file(os.path.join(agent_files_path, "MasterAgent.json"), MASTER_AGENT_PROMPT)
  initialize_agent_file(os.path.join(agent_files_path, "CodeAgent.json"), CODE_AGENT_PROMPT)
  initialize_agent_file(os.path.join(agent_files_path, "EnvAgent.json"), ENV_AGENT_PROMPT)
  loop_interaction()


# initialize the main function
if __name__ == "__main__":
    main()
