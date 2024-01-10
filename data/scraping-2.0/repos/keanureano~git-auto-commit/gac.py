import subprocess
import os
import argparse
import openai
import readline


def main():
    """
    Main function.
    """

    loadEnvVariables()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    diff = gitDiffStaged()

    generateCommitMessageLoop(diff)


def loadEnvVariables():
    """
    Load environment variables from a file.
    """

    # Get the absolute path of the currently executing script
    script_path = os.path.abspath(__file__)

    #   Get the directory containing the script
    env_path = f"{os.path.dirname(script_path)}\.env"

    try:
        with open(env_path) as f:
            for line in f:
                if "=" in line:
                    key, value = map(str.strip, line.split("=", 1))
                    os.environ[key] = value.replace("\\n", "\n")

    except FileNotFoundError:
        raise FileNotFoundError(
            f"{env_path} not found. Create a '.env' file and set OPENAI_API_KEY."
        )


def gitDiffStaged():
    """
    Executes the `git diff --staged` command and returns the output string.
    Raises a ValueError if the diff log is empty.
    """

    MAX_DIFF_CHARACTERS = 5000
    process = subprocess.Popen(["git", "diff", "--staged"], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    decoded_output = output.decode("utf-8")[:MAX_DIFF_CHARACTERS]

    if not decoded_output.strip():
        raise ValueError("No staged changes found. Did you forget to 'git add' files?")

    return decoded_output


def generateCommitMessageLoop(diff):
    """
    Generate commit messages in a loop based on user input.
    """
    while True:
        commitMessage = generateCommitMessage(diff)
        print("\nGenerated Commit Message:\n", commitMessage)

        action = input(
            "Do you want to commit this message? (Y)es, (N)o, (R)etry, (E)dit: "
        ).upper()

        if action == "Y":
            gitCommit(commitMessage)
            break
        elif action == "N":
            break  # Exit the loop and program
        elif action == "R":
            continue  # Retry the loop
        elif action == "E":
            editedMessage = inputWithPrefill(commitMessage)
            gitCommit(editedMessage)
            break
        else:
            print("Invalid option. Please enter 'Y', 'N', 'R', 'E'.")


def generateCommitMessage(diff):
    """
    Generate a commit message using GPT with the specified instruction, model, and diff logs.
    Returns the generated commit message.
    """

    instruction = selectGptInstruction()
    model = selectGptModel()

    prompt = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": diff},
    ]

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        max_tokens=150,
    )

    return response.choices[0].message.content


def gitCommit(commit_message):
    """
    Executes git commit command with the given commit message.
    """
    try:
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during Git operations: {e}")


def selectGptInstruction():
    """
    Get the GPT instruction based on the specified mode.
    Raises an error if BASE_INSTRUCTION or PROMPT_ENGINEERED_INSTRUCTION is missing or empty.
    """

    mode = getUserMode()
    base_instruction = os.getenv("BASE_INSTRUCTION")
    prompt_engineered_instruction = os.getenv("PROMPT_ENGINEERED_INSTRUCTION")

    if mode in ["base", "finetuned"]:
        if not base_instruction or not base_instruction.strip():
            raise ValueError(
                f"{mode.upper()}_INSTRUCTION not found or empty in the .env file."
            )
        return base_instruction

    if mode in ["engineered", "combined"]:
        if (
            not prompt_engineered_instruction
            or not prompt_engineered_instruction.strip()
        ):
            raise ValueError(
                f"{mode.upper()}_INSTRUCTION not found or empty in the .env file."
            )
        return prompt_engineered_instruction

    raise ValueError(
        "Invalid mode. Supported modes are 'base', 'finetuned', 'engineered', and 'combined'."
    )


def selectGptModel():
    """
    Get the model based on the specified mode.
    Throws an error if BASE_MODEL or FINETUNED_MODEL is missing.
    """

    mode = getUserMode()
    base_model = os.getenv("BASE_MODEL")
    fine_tuned_model = os.getenv("FINETUNED_MODEL")

    if mode in ["base", "engineered"]:
        if not base_model or not base_model.strip():
            raise ValueError(
                f"{mode.upper()}_MODEL not found or empty in the .env file."
            )
        return base_model

    if mode in ["finetuned", "combined"]:
        if not fine_tuned_model or not fine_tuned_model.strip():
            raise ValueError(
                f"{mode.upper()}_MODEL not found or empty in the .env file."
            )
        return fine_tuned_model

    raise ValueError(
        "Invalid mode. Supported modes are 'base', 'engineered', 'finetuned', and 'combined'."
    )


def getUserMode():
    """
    Get the user mode based on command-line arguments.
    Defaults to "base" if no valid mode is specified.
    """
    parser = argparse.ArgumentParser(description="Process GPT user mode.")

    # Define the mapping of argument names to modes and their short versions
    mode_mapping = {
        "base": ["-b", "--base"],
        "engineered": ["-e", "--engineered"],
        "finetuned": ["-f", "--finetuned"],
        "combined": ["-c", "--combined"],
    }

    # Add arguments to the parser dynamically
    for mode, arg_names in mode_mapping.items():
        parser.add_argument(
            *arg_names, dest=mode, action="store_true", help=f"Use the {mode} mode."
        )

    args = parser.parse_args()

    # Find the first mode that the user specified
    for mode in mode_mapping:
        if getattr(args, mode):
            return mode

    # Default to "base" if no valid mode is specified
    return "base"


def inputWithPrefill(preFilledText):
    def preInputHook():
        readline.insert_text(preFilledText)
        readline.redisplay()

    readline.set_pre_input_hook(preInputHook)
    userInput = input()
    readline.set_pre_input_hook()
    return userInput


if __name__ == "__main__":
    main()
