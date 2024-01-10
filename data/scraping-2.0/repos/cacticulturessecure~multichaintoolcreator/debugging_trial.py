# Required imports
import os
import subprocess
import openai
import ast
from utils import create_unique_file

# Read the OpenAI API key
with open('openai_api_key.txt', 'r') as file:
    openai.api_key = file.read().strip()

# Function to run a Python script and capture its output
def run_program_and_get_output(filename):
    result = subprocess.run(['python', filename], capture_output=True, text=True)
    return result.stdout

# Function to send the output of a Python script to the OpenAI GPT-3 model
def pass_output_to_openai(output):
    response = openai.Completion.create(engine="text-davinci-002", prompt=output, max_tokens=60)
    return response.choices[0].text.strip()

# Function to generate a task list for creating a Python tool
def tool_creation_task_list(processed_output):
    tree = ast.parse(processed_output)
    tasks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            tasks.append(f"Create function '{node.name}'")
        elif isinstance(node, ast.ClassDef):
            tasks.append(f"Create class '{node.name}'")
    return tasks

# Main execution
if __name__ == "__main__":
    # Prompt the user to enter the name of the Python file to run
    filename = input("Enter the name of the Python file to run: ")

    # Run the Python file and capture its output
    output = run_program_and_get_output(filename)

    # Send the output to the OpenAI GPT-3 model
    processed_output = pass_output_to_openai(output)

    # Generate a task list from the processed output
    tasks = tool_creation_task_list(processed_output)

    # Print the task list
    print("\nTask List for Creating a Python Tool:")
    for task in tasks:
        print(f"- {task}")