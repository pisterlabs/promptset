import sys
import os
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents import load_tools
from langchain.utilities import BashProcess

bash = BashProcess()

def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 assist.py <command_line_input>")
        sys.exit(1)

    command_line_input = sys.argv[1]
    additional_prompts = read_file_content("additional_prompts.txt")
    hardcoded_string = "You have access to the terminal through the bash variable. "

    if os.path.exists("specific_task.txt"):
        with open("specific_task.txt", "r") as f:
            specific_task = f.read()

    prompt = hardcoded_string + additional_prompts + specific_task + command_line_input
    
    agent_executor = create_python_agent(
        llm=OpenAI(temperature=0, max_tokens=1000),
        tool=PythonREPLTool(),
        verbose=True
    )
    
    agent_executor.run(prompt)

if __name__ == "__main__":
    main()

