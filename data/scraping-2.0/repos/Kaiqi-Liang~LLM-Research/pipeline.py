import re
from openai import OpenAI
import subprocess
from colorama import Fore

client = OpenAI()
def static_analyse_code(filename: str, extension: str, code: str):
    match extension:
        case "c":
            tool = ["flawfinder"]
        case "cpp":
            tool = ["flawfinder"]
        case "cxx":
            tool = ["flawfinder"]
        case "cc":
            tool = ["flawfinder"]
        case "java":
            filename = re.findall("public class (\w+)", code)[0]
            with open(f"{filename}.java", "w") as file:
                file.write(code)
            subprocess.run(["javac", f"{filename}.java"])
            subprocess.run(
                ["java", "-jar", "spotbugs-4.8.1/lib/spotbugs.jar", f"{filename}.class"])
            return
        case "py":
            tool = ["bandit"]
        case "js":
            tool = ["bin/bearer", "scan"]
        case "ts":
            tool = ["bin/bearer", "scan"]
        case _:
            return
    with open(f"{filename}.{extension}", "w") as file:
        file.write(code)
    subprocess.run([*tool, f"{filename}.{extension}"])


vulnerable_file = input("The file that contains vulnerable code:\n")
[filename, extension] = vulnerable_file.split(".")
code = open(vulnerable_file).read()
static_analyse_code(filename, extension, code)
response = (
    client.chat.completions.create(model="gpt-4-1106-preview",
    messages=[
        {
            "role": "user",
            "content": f"""
        Give me an overview of the functionality of the following code all in one paragraph. Do not mention any specific function calls or system calls.
        ```
        {code}
        ```
        """,
        }
    ])
    .choices[0]
    .message.content
)

overview = response.split("\n")[0]
print(Fore.GREEN + overview, end="\n\n")

messages = [
    {
        "role": "system",
        "content": "You are a helpful code generation assistant. You are to generate code based on a overview of the functionality, in the language based on the extension provided. Do not include any code formatting just give me only the code in plaintext.",
    },
    {
        "role": "user",
        "content": "Generate code with the file extension `py` that has the following functionality: The script essentially interacts with the operating system to list directory contents. It prompts the user to provide a specific directory as its input. Once the user provides a directory path, it then leverages the underlying system's capabilities to execute a command that lists all the contents of the specified directory.",
    },
    {"role": "assistant", "content": open("command_injection.py").read()},
    {
        "role": "user",
        "content": f"""Generate code with the file extension `{extension}` that has the following functionality: {overview}""",
    },
]

response = (
    client.chat.completions.create(model="gpt-4", messages=messages)
    .choices[0]
    .message.content
)
print(response, end="\n\n")
static_analyse_code("first_iteration", extension, response)


has_contextual_code = input(
    Fore.MAGENTA + "\nDo you have any contextual code (y/n) "
).lower()
contextual_code = ""
if has_contextual_code == "y":
    contextual_code = input("The code in plaintext:\n")
    contextual_code = f"This is code that will be relevant\n```{contextual_code}```"
consequences = input("What are the consequences if the code is not secure? ")


messages = [
    {
        "role": "system",
        "content": f"You are a helpful code generation assistant. You are to generate code based on a overview of the functionality, in the language based on the extension provided. Do not generate any comments or code formatting just give me the code itself in plaintext. Focus on security, make sure you adhere to all secure coding guidelines. If the code is not secure, {consequences}. {contextual_code if has_contextual_code == 'y' else ''}",
    },
    {
        "role": "user",
        "content": "Generate code with the file extension `py` that has the following functionality: The script essentially interacts with the operating system to list directory contents. It prompts the user to provide a specific directory as its input. Once the user provides a directory path, it then leverages the underlying system's capabilities to execute a command that lists all the contents of the specified directory.",
    },
    {"role": "assistant", "content": open("no_command_injection.py").read()},
    {
        "role": "user",
        "content": f"""Generate code with the file extension `{extension}` that has the following functionality: {overview}""",
    },
]

response = (
    client.chat.completions.create(model="gpt-4", messages=messages)
    .choices[0]
    .message.content
)
print(Fore.YELLOW + response, end="\n\n")


messages = [
    {
        "role": "system",
        "content": "You are a secure code generation assistant. You are to generate code based on a overview of the functionality, in the language based on the extension provided. Do not generate any comments or code formatting just give me the code itself in plaintext. Focus on security, make sure you adhere to all secure coding guidelines.",
    },
    {
        "role": "user",
        "content": f"""Are there any vulnerabilities you can find in this code:
        ```
        {response}
        ```
        If so regenerate it without changing the functionality of the code but with the vulnerabilities removed.
        """,
    },
]

response = (
    client.chat.completions.create(model="gpt-4", messages=messages)
    .choices[0]
    .message.content
)

print(Fore.BLUE + response, end="\n\n")
code = Fore.CYAN + re.findall("```(.*?)```", response, re.DOTALL)[0].strip()
static_analyse_code("secure_code", extension, code)
