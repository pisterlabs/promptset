import difflib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import argparse
import ssl
import urllib.error
import requests
import os
import json

import os
import json

import openai
from dotenv import load_dotenv
from termcolor import cprint
from typing import List, Dict


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")
VALIDATE_JSON_RETRY = int(os.getenv("VALIDATE_JSON_RETRY", -1))
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")

# Read the system prompt
with open(PROMPT_FILE_PATH, "r") as prompt_file:
    SYSTEM_PROMPT = prompt_file.read()

# Define standard directories for TLA+ tools based on the platform
STANDARD_DIRS = {
    "Windows": "C:/Program Files/TLA+",
    "Darwin": "/Users/Shared/TLA+",
    "Linux": "/usr/local/share/TLA+",
}


def get_standard_dir():
    """
    Get the standard directory for TLA+ tools based on the platform
    """
    return STANDARD_DIRS.get(platform.system(), "/usr/local/share/TLA+")


def run_tla_spec(spec_name: str, tla_tools_path: str) -> str:
    """
    Run TLC Model Checker on the given TLA+ specification.

    Parameters:
    spec_name: The name of the TLA+ specification to run.
    tla_tools_path: The file path to the TLA+ tools.

    Returns:
    A string containing the output of the TLC Model Checker.
    """
    subprocess_args = ["java", "-cp", tla_tools_path, "tlc2.TLC", spec_name]

    try:
        result = subprocess.check_output(subprocess_args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        return error.output.decode("utf-8"), error.returncode
    return result.decode("utf-8"), 0


def json_validated_response(
    model: str, messages: List[Dict], nb_retry: int = VALIDATE_JSON_RETRY
) -> Dict:
    """
    This function is needed because the API can return a non-json response.
    This will run recursively VALIDATE_JSON_RETRY times.
    If VALIDATE_JSON_RETRY is -1, it will run recursively until a valid json
    response is returned.
    """
    json_response = {}
    if nb_retry != 0:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.5,
        )
        messages.append(response.choices[0].message)
        content = response.choices[0].message.content
        # see if json can be parsed
        try:
            json_start_index = content.index(
                "["
            )  # find the starting position of the JSON data
            json_data = content[
                json_start_index:
            ]  # extract the JSON data from the response string
            json_response = json.loads(json_data)
            return json_response
        except (json.decoder.JSONDecodeError, ValueError) as e:
            cprint(f"{e}. Re-running the query.", "red")
            # debug
            cprint(f"\nGPT RESPONSE:\n\n{content}\n\n", "yellow")
            # append a user message that says the json is invalid
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your response could not be parsed by json.loads. "
                        "Please restate your last message as pure JSON."
                    ),
                }
            )
            # dec nb_retry
            nb_retry -= 1
            # rerun the api call
            return json_validated_response(model, messages, nb_retry)
        except Exception as e:
            cprint(f"Unknown error: {e}", "red")
            cprint(f"\nGPT RESPONSE:\n\n{content}\n\n", "yellow")
            raise e
    raise Exception(
        f"No valid json response found after {VALIDATE_JSON_RETRY} tries. Exiting."
    )


def send_error_to_gpt(
    spec_path: str, error_message: str, model: str = DEFAULT_MODEL
) -> Dict:
    # Read the TLA+ specification
    with open(spec_path, "r") as f:
        spec_lines = f.readlines()

    # Assume the .CFG file has the same name as the TLA+ specification but with a .cfg extension
    cfg_path = spec_path.rsplit(".", 1)[0] + ".cfg"

    # Read the .CFG file
    with open(cfg_path, "r") as f:
        model_cfg = f.read()

    # Full spec for context
    full_spec = "".join(spec_lines)

    # Spec lines with line numbers for reference
    spec_with_lines = []
    for i, line in enumerate(spec_lines):
        spec_with_lines.append(str(i + 1) + ": " + line)
    spec_with_lines = "".join(spec_with_lines)

    # Create the prompt for the AI model
    prompt = (
        "Here is the TLA+ spec that has errors that need fixing:\n\n"
        f"{full_spec}\n\n"
        "Here is the TLA+ model requirements that need to be met for more context:\n\n"
        f"{model_cfg}\n\n"
        "Here is the TLA+ spec lines that needs fixing:\n\n"
        f"{spec_with_lines}\n\n"
        "Here is the error message:\n\n"
        f"{error_message}\n"
        "Please provide your suggested changes, and remember to stick to the "
        "exact format as described above."
    )

    # Send the prompt to the AI model
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return json_validated_response(model, messages)


def apply_changes(file_path: str, changes: List, confirm: bool = False):
    with open(file_path) as f:
        original_file_lines = f.readlines()

    # Filter out explanation elements
    operation_changes = [change for change in changes if "operation" in change]
    explanations = [
        change["explanation"] for change in changes if "explanation" in change
    ]

    # Sort the changes in reverse line order
    operation_changes.sort(key=lambda x: x["line"], reverse=True)

    file_lines = original_file_lines.copy()
    for change in operation_changes:
        operation = change["operation"]
        line = change["line"]
        content = change["content"]

        if operation == "Replace":
            file_lines[line - 1] = content + "\n"
        elif operation == "Delete":
            del file_lines[line - 1]
        elif operation == "InsertAfter":
            file_lines.insert(line, content + "\n")

    # Print explanations
    cprint("Explanations:", "blue")
    for explanation in explanations:
        cprint(f"- {explanation}", "blue")

    # Display changes diff
    print("\nChanges to be made:")
    diff = difflib.unified_diff(original_file_lines, file_lines, lineterm="")
    for line in diff:
        if line.startswith("+"):
            cprint(line, "green", end="")
        elif line.startswith("-"):
            cprint(line, "red", end="")
        else:
            print(line, end="")

    if confirm:
        # check if user wants to apply changes or exit
        confirmation = input("Do you want to apply these changes? (y/n): ")
        if confirmation.lower() != "y":
            print("Changes not applied")
            sys.exit(0)

    with open(file_path, "w") as f:
        f.writelines(file_lines)
    print("Changes applied.")


def check_model_availability(model):
    available_models = [x["id"] for x in openai.Model.list()["data"]]
    if model not in available_models:
        print(
            f"Model {model} is not available. Perhaps try running with "
            "`--model=gpt-3.5-turbo` instead? You can also configure a "
            "default model in the .env"
        )
        exit()


def find_tla_tools_path():
    # Run the locate command to find the TLA+ tools jar file
    try:
        locate_output = (
            subprocess.check_output(["locate", "tla2tools.jar"])
            .decode("utf-8")
            .split("\n")
        )
    except subprocess.CalledProcessError:
        print(
            "The locate command failed. Please make sure that the locate database is up to date."
        )
        return None

    # Filter out any empty lines or lines that don't end with "tla2tools.jar"
    jar_files = [path for path in locate_output if path.endswith("tla2tools.jar")]

    if not jar_files:
        print("Could not find the TLA+ tools jar file.")
        return None

    # Find the jar file with the latest modification time
    latest_jar_file = max(jar_files, key=os.path.getmtime)

    return latest_jar_file


def install_tla_plus(disable_ssl_verification=False):
    """Downloads and sets up TLA+."""
    print("TLA+ not found. Attempting to install...")

    # Download the tla2tools.jar file
    url = "https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar"
    tla2tools_path = os.path.join(get_standard_dir(), "tla2tools.jar")

    try:
        # Attempt to download with SSL verification
        response = requests.get(url, verify=not disable_ssl_verification)
        response.raise_for_status()
        with open(tla2tools_path, "wb") as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during download: {e}")
        return None

    print("TLA+ has been downloaded.")

    return tla2tools_path


def check_tla_tools_availability():
    """
    Checks if the TLC model checker is available in the system path and functioning correctly.
    """

    # The name of the TLA+ tools jar file
    tla_tools_jar = "tla2tools.jar"

    # Check if TLA+ tools are available in system PATH
    tla_tools_path = shutil.which(tla_tools_jar)

    # If not found in PATH, check in our standard directory
    if not tla_tools_path:
        tla_tools_path = os.path.join(get_standard_dir(), tla_tools_jar)

        # If not found in standard directory, attempt to install
        if not os.path.isfile(tla_tools_path):
            tla_tools_path = install_tla_plus()

    # If still not found, use the locate command to find the most recent jar file
    if not tla_tools_path:
        tla_tools_path = find_tla_tools_path()

    # Define a simple TLA+ spec with a bound on x
    trivial_spec = """
    --------------------------- MODULE TempTestSpec ---------------------------
    EXTENDS Naturals
    VARIABLE x
    Init == x = 0
    Next == IF x < 5 THEN x' = x + 1 ELSE x' = x
    ==============================================================================
    """

    # Write the spec to a temporary file
    with open("TempTestSpec.tla", "w") as f:
        f.write(trivial_spec)

    # Define a simple TLA+ config
    trivial_config = """
    INIT Init
    NEXT Next
    """

    # Write the config to a temporary file
    with open("TempTestSpec.cfg", "w") as f:
        f.write(trivial_config)

    # Run TLC on the temporary spec
    try:
        result = subprocess.run(
            ["java", "-cp", tla_tools_path, "tlc2.TLC", "TempTestSpec"],
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print(
            "TLA+ Tools are not available. Please follow these steps to install:\n"
            "1. Download the TLA+ tools JAR file from https://github.com/tlaplus/tlaplus/releases\n"
            "2. Place the JAR file in a directory, such as /Users/yourusername/Dev/TLA+\n"
            "3. Make sure Java is installed on your system. You can download Java from https://www.java.com/en/download/\n"
            "4. Ensure that the TLA+ tools are accessible in your Java classpath.\n"
        )
        sys.exit(1)

    # If there was an error running TLC, print the error and exit
    if result.returncode != 0:
        print(
            "There was an error running the TLC model checker. Error details:\n"
            f"{result.stderr.decode('utf-8')}"
        )
        sys.exit(1)

    # Clean up the temporary spec and config files
    os.remove("TempTestSpec.tla")
    os.remove("TempTestSpec.cfg")

    return tla_tools_path


def provide_detailed_comments(spec_path: str, model: str = DEFAULT_MODEL):
    cfg_path = os.path.splitext(spec_path)[0] + ".cfg"

    # Step 1: Load both the .cfg and .tla files and send them in a prompt
    cfg_file_content = ""
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_file_content = f.read()

    with open(spec_path, "r") as f:
        spec_lines = f.readlines()

    prompt = (
        "Here is a TLA+ specification:\n\n"
        "tla\n"
        "".join(spec_lines) + "\n"
        "\n\n"
        "And here is its .cfg file (the model to check):\n\n"
        "Here is the .cfg file \n" + cfg_file_content + "\n"
        "\n\n"
        "Please rewrite the TLA+ file to include detailed comments "
        "that are readable and useful for developers and people new to TLA+. "
        "Ensure the code is enclosed within three backticks (```)."
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    # Step 3: Take the response and pull the TLA+ file and rewrite the file
    # We expect the TLA+ code to be in a code block in the response
    response_lines = response.choices[0].message["content"].split("\n")
    response_code_start = next(
        i for i, line in enumerate(response_lines) if line.startswith("```")
    )
    response_code_end = next(
        i
        for i, line in enumerate(
            response_lines[response_code_start + 1 :], response_code_start + 1
        )
        if line.startswith("```")
    )
    response_code = response_lines[response_code_start + 1 : response_code_end]
    response_code = [line.replace("\\\\", "\\") for line in response_code]

    # Step 4: Run a sanitation check on the TLA+ file that there are not comments on the header line
    if response_code[0].startswith("----------------------------- MODULE"):
        response_code[0] += "\n\n"

    # Step 5: Make sure there are no generic or empty comments
    response_code = [
        line
        for line in response_code
        if "\\*" not in line
        or line.split("\\*")[1].strip() not in {"", "this is a specification"}
    ]

    # Step 6: Check there are no comments in the footer
    footer_index = next(
        (i for i, line in enumerate(response_code) if line.startswith("====")), None
    )
    if footer_index is not None:
        response_code = response_code[:footer_index] + [
            line for line in response_code[footer_index:] if "\\*" not in line
        ]

    # Step 7: Write the sanitized lines back to the file
    with open(spec_path, "w") as f:
        f.write("\n".join(response_code))

    print(
        "\nDetailed comments for the specification have been added to the TLA+ file.\n"
    )


def main(spec_name, revert=False, model=DEFAULT_MODEL, confirm=False):
    if revert:
        backup_file = spec_name + ".bak"
        if os.path.exists(backup_file):
            shutil.copy(backup_file, spec_name)
            print(f"Reverted changes to {spec_name}")
            sys.exit(0)
        else:
            print(f"No backup file found for {spec_name}")
            sys.exit(1)

    tla_tools_path = check_tla_tools_availability()

    while True:
        output, returncode = run_tla_spec(spec_name, tla_tools_path)

        if returncode != 0:
            print(
                f"An error occurred when checking {spec_name} with TLC. Error message:\n{output}"
            )

            # Make a backup of the spec file
            shutil.copy(spec_name, spec_name + ".bak")

            # Send error to GPT and get suggestions
            changes_suggestion = send_error_to_gpt(spec_name, output, model)

            # Apply the changes
            apply_changes(spec_name, changes_suggestion, confirm)

            print(
                f"Changes applied to {spec_name}. Please check the spec and rerun the tool if necessary."
            )

        else:
            print(f"No errors detected in {spec_name}.")
            print("Getting detailed comments for the specification...")
            provide_detailed_comments(spec_name, model)

            # Run TLC again to check if the comments introduced any errors
            print("Running TLC again to check if the comments introduced any errors...")
            output, returncode = run_tla_spec(spec_name, tla_tools_path)
            if returncode != 0:
                print(
                    f"An error occurred after adding comments to {spec_name}. Error message:\n{output}"
                )
                print("Removing comments...")
                shutil.copy(spec_name + ".bak", spec_name)
            else:
                print("No errors detected after adding comments. Exiting...")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spec_name", type=str, help="The name of the TLA+ spec to check"
    )
    parser.add_argument(
        "--revert", action="store_true", help="Revert the spec to its previous state"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="The name of the GPT model to use",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Ask for confirmation before each change"
    )

    args = parser.parse_args()

    main(args.spec_name, args.revert, args.model, args.confirm)
