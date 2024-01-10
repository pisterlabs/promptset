import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"
TEMP = 0.5
MAX_TOKENS = 1000

SYSTEM_LOGIC = "You are an expert CLI Assistant AI for the Bash scripting language. Your goal is to, given the input request, generate a CLI command or series of CLI commands that will satisfy the request. Respond in great detail and with as much information as possible."

SYSTEM_SELECT = """You are an expert CLI Assistant AI who has already decided on a single (or series of) command(s) to use. Please, given your past reasoning, isolate out the commands you have decided on and return them without any additional information, and without wrapping them in code blocks.

In the case of multiple commands, please separate them with a semicolon (;).
"""

SYSTEM_ONESHOT = "You are an expert CLI Assistant AI for the Bash scripting language. Your goal is to, given the input request, generate a single CLI command that will satisfy the request. Respond with the command you have deemed appropriate and return only that command, making sure to not wrap it in a code block."


class CLIHelperClient:
    def get_command(self, text):
        return CLIHelperClient.get_llm_response(text, SYSTEM_ONESHOT)

    def ask_for_command(self, text):
        rationale = CLIHelperClient.get_llm_response(text, SYSTEM_LOGIC)
        commands = CLIHelperClient.get_llm_response(rationale, SYSTEM_SELECT).split(";")

        return (rationale, commands)

    @staticmethod
    def get_llm_response(text, system):
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "assistant",
                    "content": text,
                },
            ],
            temperature=TEMP,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()  # type: ignore
