import os
import re
from typing import Optional
from pathlib import Path
from langchain_g4f import G4FLLM
from g4f import ChatCompletion, models


class CodeImprover:
    """
    A pragmatic, procedural, and organized code analyzing, improving and upgrading agent.
    """
    version: Optional[int] = 2.0
    script_path: Optional[str] = None
    log_file: Optional[str] = "changes.txt"

    def __init__(self):
        self.python_mark: Optional[str] = r"```(python|py|)\n(?P<code>[\s\S]+?)\n```"

    @staticmethod
    def read_code(text: str, code_mark: str) -> str:
        match = re.search(code_mark, text)
        if match:
            return match.group("code")
        
    def save_changes(self, text: str):
        with open(self.log_file, "a") as file:
            file.write(text)
            print(f"changes saved to log {self.log_file}")

    def improve_code(self) -> None:
        """ Improves the code in a given file. """
        if self.script_path and os.path.exists(self.script_path):
            path = self.script_path
        else:
            path = input("Enter the path of the file you want to improve: ")

        try:
            with open(path, "r") as file:
                code = file.read()
            
            #protected
            prompt = f"""
            You are a pragmatic, procedural, and organized code analyzing and improving agent. 
            You can ingest, analyze, improve and upgrade scripts.
Rules:
            - Don't make a new comment block at the end of the script if one exists, just add or subtract info as needed.
            - Don't change methods, classes or variables that have '#protected' above the definition.
            - Don't remove anything, only add functionality.
            - Don't remove imports of the script.
            - Don't add any type hints to kwargs.
            - Don't remove license comments.
            - Always Update the version number by adding 0.1 to it.
            - Always list the 'TODOS:' in the comments block at the end the script.
            - Always list the changes you made under 'I made the following changes:' at the end of your response.
            - Always put a 'comment block' at the end of the script formatted like:

                Description:
                    <here the assistant describes script working>

                Usage:
                    <here the assistant describes script usage>

                Predicted use cases:
                    <here the assistant describes use cases>

                Proposed features:
                    <here the assistant proposes features>

                TODOS:
                    <here the user places todos he wants implemented>

            Tasks:
            - Refactor the code into a callable class.
            - Add a __name__ == "__main__".
            - Add variables, methods, enums, classes and logic only if needed.
            - Enhance the script.
            - Implement the items listed under 'TODOS:'.
            - Add type hints.
            - Add 'Description:' to the 'comment block' where you describe the script
            - Add 'Usage:' description  to the 'comment block' where you explain the usage.
            - Add 'Use cases:' to the 'comment block' where you list predicted use cases.
            - Add 'Proposed features:' to the 'comment block' where you list proposed new features.
            - Add 'TODOS:' to the 'comment block' where the user will add the todo's.

            Now improve the code in this file:
            ```py
            {code}
            ```
            """

            response = []
            for chunk in ChatCompletion.create(
                model=models.gpt_35_turbo,
                messages=[{"role": "user", "content": prompt}],
                timeout=6000,
                stream=False,
            ):
                response.append(chunk)
                print(chunk, end="", flush=True)

            response = "".join(response)

            code = self.read_code(response, self.python_mark)
            if code:
                new_file_path = str(Path(path).with_name(f"{Path(path).stem}_generated_improvement{Path(path).suffix}"))
                with open(new_file_path, "w") as file:
                    file.write(code)
                    print(f"Improved code saved to {new_file_path}")

            # split of the changes and save them
            changes = response.split("I made the following changes:")
            changes = f"I made the following changes in {self.version}:\n{changes.pop()}"
            self.save_changes(changes)

        except FileNotFoundError:
            print("Invalid file path.")

if __name__ == "__main__":
    CodeImprover().improve_code()

"""
Description:
A pragmatic, procedural, and organized code analyzing, improving and upgrading agent.
This script allows you to improve the code in a given file using a language model. 
The improved code will be generated based on the provided code and saved in a new file.

Usage:
- Run the script.
- Enter the path of the file you want to improve when prompted.
- The script will use the language model to generate an improved version of the code.
- The improved code will be saved in a new file with a suffix "_generated_improvement" in the file name.

Predicted use cases:
- Improving code by analyzing and providing automated suggestions.
- Learning and understanding the improvements made by the language model.

Proposed extra features:
- Add support for specifying the language model to use.
- Add support for customizing the code improvement prompt.
- Add a method to check for code syntax errors using the compile function.
- Add a method to run unit tests for the code.
- Add a method to extract code metrics like cyclomatic complexity or code duplication.

TODOS:
- Add support for improving all files in a given directory.
- Add the version to the new file name suffix.
- Add a method to check for code syntax errors using the compile function.
- Add a method to format code using autopep8 or black.
- Add a method to generate a code coverage report.
- Add self improve mode. When path input is empty process the scripts own path.
"""