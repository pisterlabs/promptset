from pathlib import Path

import openai
from cryptography.fernet import Fernet

genscript = """
checks:
  - name: "A human-readable name for the check, describing its purpose or objective. For example, 'Class Definition' or 'Function Naming Convention'."
    code: "A short, unique identifier or code that distinguishes the check from others. This code is often used for reference or automation."
    id: "A unique identifier or tag that provides additional context to the check. It may include information such as the check's category or origin. For instance, 'C001' could represent a 'Code Quality' category."
    pattern: "An XPath expression specifying the exact code pattern or element that the check aims to locate within the source code. This expression serves as the search criteria for the check's evaluation."
    count:
      min: "The minimum number of times the specified pattern is expected to appear in the source code. It establishes the lower limit for the check's results, ensuring compliance or a minimum level of occurrence."
      max: "The maximum number of times the specified pattern is expected to appear in the source code. It defines an upper limit for the check's results, helping to identify potential issues or excessive occurrences."

Example:

  - name: "class-definition"
    code: "CDF"
    id: "C001"
    pattern: './/ClassDef'
    count:
      min: 1
      max: 50
  - name: "all-function-definition"
    code: "AFD"
    id: "F001"
    pattern: './/FunctionDef'
    count:
      min: 1
      max: 200
  - name: "non-test-function-definition"
    code: "NTF"
    id: "F002"
    pattern: './/FunctionDef[not(contains(@name, "test_"))]'
    count:
      min: 40
      max: 70
  - name: "single-nested-if"
    code: "SNI"
    id: "CL001"
    pattern: './/FunctionDef/body//If'
    count:
      min: 1
      max: 100
  - name: "double-nested-if"
    code: "DNI"
    id: "CL002"
    pattern: './/FunctionDef/body//If[ancestor::If and not(parent::orelse)]'
    count:
      min: 1
      max: 15

"""

API_KEY_FILE = "userapikey.txt"


def save_user_api_key(user_api_key):
    key = Fernet.generate_key()
    fernet = Fernet(key)
    encrypted_key = fernet.encrypt(user_api_key.encode()).decode()
    with open(API_KEY_FILE, "w") as f:
        f.write(key.decode() + "\n" + encrypted_key)


def load_user_api_key(file):
    with open(file, "r") as f:
        lines = f.read().strip().split("\n")
        if len(lines) == 2:  # noqa: PLR2004
            key = lines[0].encode()
            encrypted_key = lines[1]
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_key.encode()).decode()


def is_valid_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Test message"}],
        )
        return True
    except openai.error.OpenAIError:
        return False


def generate_yaml_config(file: Path, user_api_key, user_input: str) -> str:
    try:
        openai.api_key = user_api_key

        prompts = [
            genscript
            + "in the same format as what is shown above(do not just generate the example use it as a framework nothing else): "
            + user_input
        ]

        response = openai.ChatCompletion.create(  # type: ignore
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that generates YAML configurations. Your task is to {prompts}",
                }
            ],
        )

        generated_yaml = response.choices[0].message["content"].strip()
        file.touch()

        with open(file, "w") as f:
            f.write(generated_yaml)

        return generated_yaml

    except openai.error.OpenAIError:  # type: ignore
        return "[red][Error][/red] There was an issue with the API key. Make sure you input your API key correctly."
