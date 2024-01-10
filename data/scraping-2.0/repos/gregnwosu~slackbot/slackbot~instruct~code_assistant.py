from typing import List
from pydantic import Field
from instructor import OpenAISchema
import subprocess
from openai import OpenAI
from slackbot.instruct.utils import get_completion


class ExecutePyFile(OpenAISchema):
    """Run existing python file from local disc."""
    file_name: str = Field(
        ..., description="The path to the .py file to be executed."
    )

    def run(self):
        """Executes a Python script at the given file path and captures its output and errors."""
        try:
            result = subprocess.run(
                ['python3', self.file_name],
                text=True,
                capture_output=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred: {e.stderr}"


class File(OpenAISchema):
    """
    Python file with an appropriate name, containing code that can be saved and executed locally at a later time. This environment has access to all standard Python packages and the internet.
    """
    chain_of_thought: str = Field(...,
                                  description="Think step by step to determine the correct actions that are needed to be taken in order to complete the task.")
    file_name: str = Field(
        ..., description="The name of the file including the extension"
    )
    body: str = Field(..., description="Correct contents of a file")

    def run(self):
        with open(self.file_name, "w") as f:
            f.write(self.body)
        return "File written to " + self.file_name


def create(client: OpenAI):
    assistant = client.beta.assistants.create(
        name='Code Assistant Agent',
        instructions="""As a top-tier programming AI, you are adept at creating accurate Python scripts. 
                      You will properly name files and craft precise Python code with the appropriate 
                      imports to fulfill the user's request. Ensure to execute the necessary 
                      code before responding to the user.""",
        model="gpt-4-1106-preview",
        tools=[{"type": "function", "function": File.openai_schema},
               {"type": "function", "function": ExecutePyFile.openai_schema}, ]
    )
    return assistant
