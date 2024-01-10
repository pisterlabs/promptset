import ast
import openai
from utils import create_unique_file
import subprocess
import os  # Make sure this import is at the top of your script
import time



with open('openai_api_key.txt', 'r') as file:
    OPENAI_API_KEY = file.read().strip()
openai.api_key = OPENAI_API_KEY

color_prefix_by_role = {
    "system": "\033[0m",  # gray
    "user": "\033[0m",  # gray
    "assistant": "\033[92m",  # green
}


def print_messages(messages, color_prefix_by_role=None) -> None:
    """Prints messages sent to or from GPT, with optional custom roles and colors."""
    if color_prefix_by_role is None:
        color_prefix_by_role = {
            "system": "\033[0m",  # gray
            "user": "\033[0m",    # gray
            "assistant": "\033[92m",  # green
        }
    for message in messages:
        if "role" not in message or "content" not in message:
            print("Invalid message format: role and content required.")
            continue
        role = message["role"]
        content = message["content"]
        color_prefix = color_prefix_by_role.get(role, "\033[0m")  # Default to gray if role is unknown
        print(f"{color_prefix}\n[{role}]\n{content}")



def print_message_delta(delta, color_prefix_by_role=color_prefix_by_role) -> None:
    """Prints a chunk of messages streamed back from GPT."""
    if "role" in delta:
        role = delta["role"]
        color_prefix = color_prefix_by_role[role]
        print(f"{color_prefix}\n[{role}]\n", end="")
    elif "content" in delta:
        content = delta["content"]
        print(content, end="")
    else:
        pass


def run_program_in_powershell(filename):
    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"The file {filename} does not exist.")
        return

    # Get the full path of the file
    full_path = os.path.join(os.getcwd(), filename)

    # Implement delay to ensure file is present before opening PowerShell
    time.sleep(1)

    # Construct the argument list for the PowerShell command
    args = ["powershell", "-ExecutionPolicy", "Unrestricted", full_path]

    # Run the PowerShell command
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the PowerShell command: {e}")



def tool_creation_task_list(
    task_description: str,  # Python function to test, as a string
    unit_test_package: str = "pytest",  # unit testing package; use the name as it appears in the import statement
    approx_min_cases_to_cover: int = 4,  # minimum number of test case categories to cover (approximate)
    print_text: bool = True,  # optionally prints text; helpful for understanding the function & debugging
    explain_model: str = "gpt-4-0613",  # model used to generate text plans in step 1
    plan_model: str = "gpt-4-0613",  # model used to generate text plans in steps 2 and 2b
    execute_model: str = "gpt-4-0613",  # model used to generate code in step 3
    temperature: float = 0.4,  # temperature = 0 can sometimes get stuck in repetitive loops, so we use 0.4
    reruns_if_fail: int = 1,  # if the output code cannot be parsed, this will re-run the function up to N times
) -> str:
    """Returns a task list to create a tool, using a 4-step GPT prompt."""

    # Step 1: Understanding the Task
    explain_system_message = {
        "role": "system",
        "content": "You are a seasoned Python developer and a creative problem solver. You understand complex tasks and translate them into actionable plans. Please carefully analyze the user's requirements for the Python tool they need."
    }
    explain_user_message = {
        "role": "user",
        "content": f"I need a Python tool that {task_description}. Can you help me understand how this can be implemented?"
    }
    explain_messages = [explain_system_message, explain_user_message]

    explanation_response = openai.ChatCompletion.create(
        model=explain_model,
        messages=explain_messages,
        temperature=temperature,
        stream=True,
    )
    explanation = ""
    for chunk in explanation_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            explanation += delta["content"]
    explain_assistant_message = {"role": "assistant", "content": explanation}

    # Step 2: Planning the Tool
    plan_user_message = {
        "role": "user",
        "content": "Based on the task description, please outline a detailed plan for creating the Python tool. Include the necessary modules, methods, classes, and any dependencies. Organize your plan in a logical sequence."
    }
    plan_messages = [explain_system_message, explain_user_message, explain_assistant_message, plan_user_message]
    plan_response = openai.ChatCompletion.create(
        model=plan_model,
        messages=plan_messages,
        temperature=temperature,
        stream=True,
    )
    plan = ""
    for chunk in plan_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            plan += delta["content"]
    plan_assistant_message = {"role": "assistant", "content": plan}

    # Step 3: Generating the Tool Code
    execute_system_message = {
        "role": "system",
        "content": "You are a world-class Python developer skilled in translating plans into efficient and well-structured code. Please create the Python code for the tool based on the plan provided, ensuring all requirements are met."
    }
    execute_user_message = {
        "role": "user",
        "content": "Based on the plan outlined above, please write the Python code for the tool. Include comments to explain each section of the code. Reply only with code, formatted as follows:\n```python\n# code here\n```"
    }
    execute_messages = [execute_system_message, explain_assistant_message, plan_assistant_message, execute_user_message]
    execute_response = openai.ChatCompletion.create(
        model=execute_model,
        messages=execute_messages,
        temperature=temperature,
        stream=True,
    )
    execution = ""
    for chunk in execute_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            execution += delta["content"]
    # Extract the code from the execution string
    code = execution.split("```python")[1].split("```")[0].strip()

    # Leverage the below method from utils.py script to save code produced to unique file
    create_unique_file(code)  # This line should be removed
    # Step 4: Code Validation
    try:
        ast.parse(code)
    except SyntaxError as e:
        error_message = f"Syntax error in generated code: {e}"
        if print_text:
            print(error_message)
        return error_message

    # Step 5: Run the Python script in a new PowerShell window
    # Leverage the below method from utils.py script to save code produced to unique file
    filename = create_unique_file(code)  # This line is correct and captures the unique filename
    error_code = run_program_in_powershell(filename)

    if error_code != 0:
        print(f"Error code: {error_code}. Diagnostics needed.")
        # Here, you can add logic to handle the error code and improve the code

    # Return the generated code as a string
    return code
