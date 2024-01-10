from openai import OpenAI
import subprocess
import sys
import platform
import time

rules = f"""
Rules:
- If this is possible in any way using Python, respond ONLY with the Python code needed to perform the action.
- If a dependency (like a pip package) needs to be installed before running the generated code, add a comment saying # !!! <command to run to set up the dependency> at the very top of the generated source.
- If package needs to be installed then # !!! <command to run to set up the dependency> is the ONLY way you can explain this, there is no need for additional comments or explanations. It is forbidden to change this format.
- ALWAYS USE extra pip packages if this will make the code smaller, simpler or faster.
- Make sure your code is correct and functions as requested. Write comments where it’s reasonably necessary.
- The target operating system for the code is {platform.platform()}.
- The target python version is {platform.python_version()}.
- It's important to wrap the code with the python code markdown tag.
- It's forbidden to produce more than one python code block. You must put everything into a single python markdown block.
- If a custom asset is requested, for example, a sample picture - it's forbidden to write a code that looks for a sample file. If sample file needed then create one.
"""


def print_gpt(text):
    print(f"\033[93m{text}", end="\033[93m")


def execute_command_line(command_line):
    log(
        f"InteractiveGPT: ⚠️  Executing: {command_line}",
        "yellow",
    )
    try:
        subprocess.check_call(command_line.split(" "))
    except Exception as e:
        log(f"InteractiveGPT: Command line exec error occurred: {e}", "red")


def install_pending_packages(code):
    lines = code.split("\n")
    for line in lines:
        if line.startswith("# !!!"):
            command_line = line[6:]
            execute_command_line(command_line)


def install_module(module_name):
    """
    Install a Python module using pip.
    """
    log(
        f"InteractiveGPT: ⚠️  Attempting to install missing module: {module_name}",
        "yellow",
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])


def execute_string_as_code(reply):
    """
    Execute a multi-line string as Python code, handle exceptions, and
    attempt to install missing modules.
    """
    code_string = extract_python_code(reply)
    if not code_string:
        log(
            f"InteractiveGPT: No code found in GPT's response. Nothing to execute.",
            "red",
        )
        pass

    while True:
        try:
            install_pending_packages(reply)

            log("\nInteractiveGPT: 😱😱😱 Executing the code 😱😱😱 \n", "cyan")

            exec(code_string)

            log("\nInteractiveGPT: ✅ Executed", "green")
            break  # Break the loop if the code executed successfully
        except ModuleNotFoundError as e:
            # Extract the module name from the exception message
            missing_module = str(e).split("'")[1]

            install_module(missing_module)
        except Exception as e:
            log(f"InteractiveGPT: An error occurred: {e}", "red")
            break  # Exit the loop if an error other than ModuleNotFoundError occurs


def log(text, color):
    """
    Print text in the specified color to the console.
    """
    colors = {
        "standard": "\033[0m",  # Reset to default color
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "green": "\033[92m",
        "dark_gray": "\033[90m",  # Dark gray color
    }
    color_code = colors.get(color, colors["standard"])
    print(f"{color_code}{text}{colors['standard']}")


def read_string_from_file(file_name):
    """
    Read a single line from a file and handle file not found exception.
    """
    try:
        with open(file_name, "r") as file:
            return file.readline().strip()
    except FileNotFoundError:
        return "File not found."


def extract_python_code(markdown_string):
    """
    Extract Python code from a markdown string.
    """
    lines = markdown_string.split("\n")
    start_python_code = False
    python_code = []

    for line in lines:
        if line.strip().lower() == "```python":
            start_python_code = True
            continue
        if line.strip() == "```" and start_python_code:
            break
        if start_python_code:
            python_code.append(line)

    return "\n".join(python_code)


def main():
    """
    Main function to run the interactive GPT interface.
    """
    token = read_string_from_file("token")
    if token == "File not found.":
        log("Error: provide your OpenAI token in ./token file", "red")
        sys.exit(1)

    gpt_model = "gpt-4"

    if len(sys.argv) > 1:
        gpt_model = sys.argv[1]

    log("InteractiveGPT: Hey! I will turn your requests to code and execute.", "cyan")
    log("InteractiveGPT: ⚠️  This program executes AI-generated code.", "yellow")
    log(
        "InteractiveGPT: ⚠️  YOU are solely responsible for understanding and accepting the risks associated with its use.",
        "yellow",
    )
    log(
        "InteractiveGPT: Source code: https://github.com/dz333n/interactive-gpt",
        "dark_gray",
    )
    log(
        f"InteractiveGPT: Using {gpt_model}, platform: {platform.platform()}, python: {platform.python_version()}",
        "dark_gray",
    )

    while True:
        user_input = input("\n> ")
        if user_input.lower() == "exit":
            log("Bye", "cyan")
            break

        process_user_input(user_input, gpt_model, token)


def process_user_input(user_input, gpt_model, token):
    """
    Process user input and interact with the GPT model.
    """
    start_time = time.time()
    prompt = f'Write Python code to perform the following task: "{user_input}"\n{rules}'
    # log(f"\n[Generated Prompt]\n{prompt}", "dark_gray")
    log("InteractiveGPT: ⌚ Processing your prompt...", "cyan")

    openai = OpenAI(api_key=token)
    stream = openai.chat.completions.create(
        model=gpt_model, messages=[{"role": "user", "content": prompt}], stream=True
    )

    reply = ""

    print_gpt("\nChatGPT:\n")

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            part = chunk.choices[0].delta.content
            reply += part
            print_gpt(part)

    print()

    end_time = time.time()
    time_taken = end_time - start_time
    log(f"InteractiveGPT: Processed in ⌚ {time_taken:.2f} seconds", "cyan")

    execute_string_as_code(reply)


if __name__ == "__main__":
    main()
