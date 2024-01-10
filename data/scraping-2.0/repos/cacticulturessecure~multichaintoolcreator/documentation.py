import ast
import openai
from utils import create_unique_file
import subprocess
import os  # Make sure this import is at the top of your script




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
    try:
        # Combine the current working directory with the filename to get the full path
        full_path = os.path.join(os.getcwd(), filename)

        # Create a temporary file to store the output
        temp_output_file = os.path.join(os.getcwd(), 'temp_output.txt')

        # Construct the argument list as a single string
        argument_list = f'"python {full_path} > {temp_output_file}"'

        # Command to open PowerShell and run the Python script, redirecting output to the temporary file
        command = ['powershell', 'Start-Process', 'powershell', '-ArgumentList', argument_list, '-Wait']

        # Run the command and wait for completion
        subprocess.run(command, check=True)

        # Read the output from the temporary file
        with open(temp_output_file, 'r') as file:
            output = file.read()

        # Print the output and optionally save it to another file
        print("Program ran successfully")
        print(output)

        # Define the output file name
        output_file_name = 'final_output.txt'

        # Open the file in write mode, creating it if it doesn't exist
        with open(output_file_name, 'w') as file:
            # Write the output to the file
            file.write(output)

        print(f"Output has been saved to {output_file_name}")

        # Remove the temporary output file
        os.remove(temp_output_file)

        return 0

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return -1



def generate_tool_documentation(
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

    # Step 1: Understanding the Code
    understand_system_message = {
        "role": "system",
        "content": "You are a skilled Python analyst with a deep understanding of code. Please carefully analyze the given Python code to understand its purpose, functions, and components."
    }
    understand_user_message = {
        "role": "user",
        "content": f"Here is the Python code I want you to analyze:\n```python\n{task_description}\n```"
    }
    understand_messages = [understand_system_message, understand_user_message]
    understanding_response = openai.ChatCompletion.create(
        model=explain_model,
        messages=understand_messages,
        temperature=temperature,
        stream=True,
    )
    understanding = ""
    for chunk in understanding_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            understanding += delta["content"]
    understand_assistant_message = {"role": "assistant", "content": understanding}

    # Step 2: Explaining the Code
    explain_user_message = {
        "role": "user",
        "content": "Can you provide a detailed explanation of how this code works? Include insights into the logic, functions, and any specific details that make this code unique."
    }
    explain_messages = [understand_system_message, understand_assistant_message, explain_user_message]
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

    # Step 3: Listing Use Cases
    use_cases_user_message = {
        "role": "user",
        "content": "What are the potential use cases for this code? How can an agent utilize this code for various tasks? Please provide a bulleted list of possible applications."
    }
    use_cases_messages = [understand_system_message, understand_assistant_message, explain_assistant_message,
                          use_cases_user_message]
    use_cases_response = openai.ChatCompletion.create(
        model=explain_model,
        messages=use_cases_messages,
        temperature=temperature,
        stream=True,
    )
    use_cases = ""
    for chunk in use_cases_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            use_cases += delta["content"]

    # Combining the understanding, explanation, and use cases
    final_output = f"{understanding}\n{explanation}\n{use_cases}"

    # Return the final output as a string
    return final_output