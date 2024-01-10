import os
import openai
import subprocess
import sys

# Configure the OpenAI API
openai.api_base = "http://localhost:8080/v1"
openai.api_key = ("sk-1234567890")

headers = {
    "Content-Type": "application/json",
}
data = {
    "model": "orca-mini-7b.ggmlv3.q4_0.bin",
    "prompt": prompt,
    "temperature": 0.2
}

def get_code_from_openai(prompt):
    response = openai.Completion.create(
        engine="text-davinci-004",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

def execute_code(code, timeout=10):
    with open("temp_code.py", "w") as f:
        f.write(code)

    try:
        result = subprocess.run([sys.executable, "temp_code.py"], capture_output=True, text=True, check=True, timeout=timeout)
        return result.stdout, False
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr, True
    except subprocess.TimeoutExpired:
        return "Execution timed out.", True

def request_new_feature(code):
    prompt = f"The following Python code is working without errors:\n\n{code}\n\n Please generate a brand new feature that is different from the existing code and provide the Python code for it. Write a concise feature scope in comments before the new code."
    new_feature_code = get_code_from_openai(prompt)
    return new_feature_code

def main():
    previous_feature = ""
    while True:
        prompt = "Write code for a python program that finds funny things on the internet and adds them to a CSV"
        error_exists = True
        while error_exists:
            print("Generating code using OpenAI API...")
            # Generate code using OpenAI API
            code = get_code_from_openai(prompt)
            print("Executing the code and checking for errors...")

            # Execute the code and capture the output
            output, error_exists = execute_code(code)
            if error_exists:
                print("Errors found, sending output to GPT-4 for fixing...")
                # Send the output to GPT-4 to fix the errors
                prompt = f"The following Python code has some errors:\n\n{code}\n\nError message:\n{output}\n\nPlease fix the errors and provide the corrected code."
        while not error_exists:
            print("No errors found. Requesting a new feature...")
            # When there are no errors, ask GPT to suggest a new feature
            new_feature = request_new_feature(code)
            print("Adding new feature to the code and checking for errors...")

            # Add the new feature to the code and check for errors again
            code += "\n\n" + new_feature
            output, error_exists = execute_code(code)
            if error_exists:
                print("Errors found in the new feature, sending output to GPT-4 for fixing...")
                # Send the output to GPT-4 to fix the errors in the new feature
                prompt = f"The following Python code has some errors after adding the new feature:\n\n{code}\n\nError message:\n{output}\n\nPlease fix the errors and provide the corrected code."

if __name__ == "__main__":
    main()