import openai
import json
from getpass import getpass

class GPTScriptCraft:
    def __init__(self):
        self.api_key = getpass("Please enter your OpenAI API key: ")
        openai.api_key = self.api_key

    def generate_code(self, prompt, tokens=200):
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            temperature=0.5,
            max_tokens=tokens
        )
        return response.choices[0].text.strip()

    def get_user_input(self):
        script_type = input("Enter the type of script you want to generate: ")
        purpose = input("Enter the purpose of the script: ")
        parameters = input("Enter the desired parameters (comma-separated): ")
        return script_type, purpose, parameters

    def customize_code(self, code):
        variables = input("Enter any variables you want to specify (comma-separated): ")
        function_names = input("Enter any function names you want to specify (comma-separated): ")
        for variable in variables.split(','):
            code = code.replace(f"var_{variable}", variable)
        for function in function_names.split(','):
            code = code.replace(f"func_{function}", function)
        return code

    def run(self):
        script_type, purpose, parameters = self.get_user_input()
        prompt = f"Write a Python script of type '{script_type}' for the purpose of '{purpose}' with parameters '{parameters}'"
        code = self.generate_code(prompt)
        code = self.customize_code(code)
        print("\nGenerated Python Code:\n")
        print(code)


if __name__ == "__main__":
    script_craft = GPTScriptCraft()
    script_craft.run()