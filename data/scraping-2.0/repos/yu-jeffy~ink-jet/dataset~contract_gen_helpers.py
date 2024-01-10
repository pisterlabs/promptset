import openai
from dotenv import load_dotenv
import os
import subprocess


###############################################
# HELPER FUNCTIONS
###############################################

# gpt call
def generate_smart_contract(prompt):
    # Load openai api key
    load_dotenv()
    # Create the chat completion
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=3000,  # larger token size to fit full smart contract
    )
    content = response.choices[0].message.content
    # print(content)
    return content


# Removes ```rust from the beginning and ``` from the end of the string (gpt response).
def remove_mardown_markers(text):
    # Check if the string starts with ```rust and ends with ```
    if text.startswith("```rust") and text.endswith("```"):
        # Remove ```rust from the beginning (7 characters) and ``` from the end (3 characters)
        return text[7:-3]
    else:
        # Return the original string if it doesn't have the specific markers
        return text


# Create a new cargo contract project
def create_cargo_contract_project(folder_name):
    return subprocess.run(["cargo", "contract", "new", folder_name], capture_output=True, text=True)


# Write your Rust code to the lib.rs file in the new project folder
def write_to_lib_rs(folder_name, rust_code):
    lib_rs_path = os.path.join(folder_name, "lib.rs")
    with open(lib_rs_path, 'w') as file:
        file.write(rust_code)


# Build the cargo contract
def build_cargo_contract(folder_name):
    orig_dir = os.getcwd()
    os.chdir(folder_name)
    result = subprocess.run(["cargo", "contract", "build"], capture_output=True, text=True)
    os.chdir(orig_dir)
    return result


# Run Coinfabrik Scout
def run_coinfabrik_scout(folder_name):
    orig_dir = os.getcwd()
    os.chdir(folder_name)
    result = subprocess.run(["cargo", "scout-audit", "--output-format", "json"], capture_output=True, text=True)
    os.chdir(orig_dir)
    return result


# Write 'success' or errors to a file in the project folder
def write_build_result_to_file(folder_name, result):
    result_file_path = os.path.join(folder_name, "build_result.txt")
    with open(result_file_path, 'w') as file:
        if result.returncode == 0:
            file.write("success\n")
        else:
            file.write(result.stdout)
            file.write(result.stderr)


# Write Coinfabrik Scout run results to a file in the project folder
def write_audit_result_to_file(folder_name, result):
    result_file_path = os.path.join(folder_name, "audit_result.txt")
    with open(result_file_path, 'w') as file:
        if result.returncode == 0:
            file.write("success\n")
        else:
            file.write(result.stdout)
            file.write(result.stderr)
