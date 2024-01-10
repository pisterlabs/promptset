import os
import openai
import subprocess
from pathlib import Path
import re

import dotenv
dotenv.load_dotenv()

assert os.getenv(
    "OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY in .env file"

# Please replace `your-openai-api-key` with your own OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
engine = "gpt-3.5-turbo"
max_tokens = 2048
feedback = ""

def extract_python_code(s: str) -> str:
    pattern = r'```python(.*?)```'
    match = re.search(pattern, s, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""
    

def create_openai_chat_response(prompt: str) -> str:
    """Create a chat response using OpenAI API."""
    global feedback
    if feedback != "":
        prompt = f"[no prose] This is what should be most important. Modify the code to follow the feedback as given: {feedback} \n{prompt} "
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=max_tokens,
    )
    completion = response["choices"][0]["message"]["content"]  # type: ignore
    completion = extract_python_code(completion)
    return completion


def generate_code_from_prompt(prompt: str, error: str = "") -> str:
    print("Generating code")
    """Generate Python code using OpenAI API based on the given prompt."""
    prompt = f"Only return python code to this problem with proper indentation and do not include any test suite in the output if given: {prompt}\nPython script: "
    if error:
        prompt = f"{prompt}\nError encountered, please write code to fix: {error}"

    return create_openai_chat_response(prompt)

def generate_interactive():
    return

def generate_test_cases_from_prompt(prompt: str) -> str:
    print("Generating test cases")
    """Generate Python test cases using OpenAI API based on the given prompt."""
    prompt = f"[no prose] {prompt}\nOnly write the Python test case code that will validate if the function works. If you provide explanations or extra statements such as 'Here is the Python test case', provide it as a comment with #. Write a main function that will run the test suite when run as the file. If the python function uses a GUI then do not write proper tests and write one test that will return true after running the function: "
    return create_openai_chat_response(prompt)

def generate_code_and_test(prompt0: str):
    print("Generating code and test cases")
    """Generate code and test cases based on the given prompt."""
    code = generate_code_from_prompt(prompt0)
    test_prompt = f"[no prose] The prompt: {prompt0} \nThe code: {code}"
    test_cases = generate_test_cases_from_prompt(test_prompt)
    return code, test_cases

def generate_unicode_emoji_from_prompt(prompt: str) -> str:
    """
    Generate a Unicode representation for an emoji based on the given prompt.
    """
    prompt_for_emoji = f"[no prose] Output a single suitable emoji (in Unicode format) for an applet that {prompt}? Only output Unicode, NOTHING ELSE. "
    emoji_suggestion = create_openai_chat_response(prompt_for_emoji)
    print(emoji_suggestion)
    
    # Convert the suggested emoji to its Unicode representation.
    # If the result is not a valid emoji or if the response is not clear, default to U+2699 (⚙️ - gear).
    try:
        unicode_emoji = "U+" + "-".join([f"{ord(char):X}" for char in emoji_suggestion])
        # A simple check for valid Unicode emojis can be done based on length or other criteria.
        if len(emoji_suggestion) > 2:
            return "U+2699"  # Default to ⚙️ (gear) 
        return unicode_emoji
    except:
        return "U+2699"  # Default to ⚙️ (gear) 


def generate_file_name_from_prompt(prompt: str) -> str:
    print("Generating filename")
    """Generate file using OpenAI API based on the given prompt."""
    prompt = f"Create a short python file name with the file extension for this task: {prompt}. DO NOT INCLUDE PROSE. "
    filename = create_openai_chat_response(prompt)
    print("Filename: ", filename)
    
    # Check if filename is blank and provide a fallback
    if not filename.strip():
        print("OpenAI API returned an empty filename. Using a default filename. ")
        filename = "generated_code.py"
        
    return filename


def write_code_to_file(code: str, filename: str):
    print("[no prose] Writing code to file")
    """Write the given code to a file with the given filename."""
    with open(filename, 'w') as f:
        f.write(code)


def run_code(filename: str) -> str:
    print("Running code")
    """Run a Python code file and return the error output."""
    process = subprocess.Popen(
        ['py', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    return stderr.decode()

def get_user_feedback():
    """Get user feedback on the generated code."""
    user_input = input("Are you happy with this applet? (y/n): ")
    if user_input.lower() == "y":
        print("Great! Your applet has been saved.")
        delete_files('test.py', 'checker.py')
        return ""
    elif user_input.lower() == "n":
        feedback = input("Please entery any feedback.\nFeedback: ")
        return feedback
    else:
        return get_user_feedback()
    
def save_and_run_tests(code: str, test_cases: str, title: str):
    """Save the generated code and test cases to files and run the tests."""
    write_code_to_file(code, os.path.join(save_dir, title))
    write_code_to_file(test_cases, 'test.py')
    test_str = f"{code} \n{test_cases}"
    write_code_to_file(test_str, 'checker.py')
    error = run_code("checker.py")
    return error, test_str

def display_code_info(code: str, test_cases: str, error: str):
    """Display the generated code, test cases, and error (if any) to the user."""
    print('=====================')
    print(f"Code: \n{code}")
    print('---------------------')
    print(f"Test Cases: \n{test_cases}")
    print('---------------------')
    print(f"Error: \n{error}")
    print('=====================')

def delete_files(*filenames):
    """Delete specified files if they exist."""
    for filename in filenames:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

def reverse_prompt(code):
    """
    Analyzes generated code for placeholders and prompts the user to provide specifics.
    Returns the code with user-provided values.
    """
    PLACEHOLDERS = {
    "email": ["example@email.com", "user@example.com"],
    "phone": ["123-456-7890", "(123) 456-7890"],
    "password": ["yourpassword"],
    # ... add other placeholders as needed
    }
    for category, placeholders in PLACEHOLDERS.items():
        for placeholder in placeholders:
            if placeholder in code:
                user_value = input(f"Please provide a value for {category}: ")
                code = code.replace(placeholder, user_value)
    return code

save_dir = Path(__file__).parent / "applets"

print(save_dir)

#TODO co.generate to sprinkle joy
def code_loop(prompt0: str = "", test_cases: str = "", error: str = "", loop: int = 0, title: str = "", user_feedback: str = "", max_loops: int = 10, is_server: bool= False):
    """Run a loop that generates Python code, writes them to files, and runs the code."""

    if loop >= max_loops:
        raise Exception("Error: Maximum loop count reached! Process terminated.")    
    loop += 1
    print(f"Loop: {loop}")

    prompt0 = prompt0 if prompt0 else input("Please enter the prompt for the task: ")
    title = title if title else generate_file_name_from_prompt(
            f"Create a short python file name with the file extension for this task: {prompt0}. If you need user-specific information, such as an email address, phone number, or password, prompt the user for more information. ") 
    unicode_emoji = generate_unicode_emoji_from_prompt(prompt0)
    error = error if error else user_feedback       
    code = generate_code_from_prompt(prompt0, error) 
    reverse_prompt(code)

    if not test_cases:
        test_prompt = f"[no prose] The prompt: {prompt0} \nThe code: {code} "
        test_cases = generate_test_cases_from_prompt(test_prompt)
    error, test_str = save_and_run_tests(code, test_cases, title)

    if error:
        try:
            print(error)
            code_loop(prompt0, test_cases, error, loop, title)
            error = ""
        except:
            return code, test_cases, False, unicode_emoji

    if is_server:
        return code, test_cases, True, unicode_emoji
    
    display_code_info(code, test_cases, error)

    if test_str:
        user_feedback = get_user_feedback()
        if user_feedback:
            code_loop(prompt0, test_cases, error, 0, title, user_feedback) 



if __name__ == "__main__":
    code_loop()
