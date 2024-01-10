import openai
import re


class GPTWhisperer:
    def __init__(self, api_key: str, os_fullname: str, shell: str):
        self.os_fullname = os_fullname
        self.shell = shell
        openai.api_key = api_key

    def _construct_prompt(self, task: str, explain: bool = False):
        explain_text = (
            "Also, provide a detailed explanation of how the command operates."
            if explain
            else ""
        )
        prompt = f"""Instructions: Compose a CLI command that accomplishes the following task: {task}. Ensure the command is accurate and compatible with {self.os_fullname} using {self.shell}. {explain_text}
        Format:
        Command: <insert_command_here>
{"Description: <insert_description_here> The description should match the user's language." if explain else ""}
Avoid enclosing the command with extra quotes or backticks."""
        return prompt

    def request(self, task: str, explain: bool = False):
        prompt = self._construct_prompt(task, explain)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a command line tool that generates CLI commands for users.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300 if explain else 200,
            temperature=0,
        )
        responses_processed = response["choices"][0]["message"]["content"].strip()

        command = re.search(r"Command: (.*)", responses_processed).group(1).strip()
        description = (
            re.search(r"Description: (.*)", responses_processed).group(1).strip()
            if explain
            else None
        )

        return command, description
