import itertools
from abc import abstractmethod
import re
import shlex
from typing import Dict, NamedTuple

from langchain.schema import BaseOutputParser

# Command response format
COMMAND_LINE_START = "<cmd>"
COMMAND_LINE_END = "</cmd>"
COMMAND_FORMAT = f"{COMMAND_LINE_START} command_name --arg1 value1 --arg2 value2{COMMAND_LINE_END}"


class GPTCommand(NamedTuple):
    name: str
    args: Dict


class BaseCommandGPTOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> GPTCommand:
        """Return GPTCommand"""


def preprocess_json_input(input_str: str) -> str:
    """
    Replace single backslashes with double backslashes,
    while leaving already escaped ones intact
    """
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
    )
    return corrected_str


class CommandGPTOutputParser(BaseCommandGPTOutputParser):
    """
    Custom Parser for CommandGPT that extracts the command string from the response and maps it to a GPTCommand.
    """

    def parse(self, text: str) -> GPTCommand:
        try:
            start_index = text.find(COMMAND_LINE_START)
            end_index = text.find(
                COMMAND_LINE_END, start_index + len(COMMAND_LINE_START))

            if start_index == -1 or end_index == -1:
                raise ValueError(
                    f"Invalid command line format. Expected '{COMMAND_LINE_START}' and '{COMMAND_LINE_END}'")

            # Extract the command string, stripping any leading/trailing whitespace or newline characters
            cmd_str = text[start_index +
                           len(COMMAND_LINE_START):end_index].strip()

            # If the command string starts with a newline, remove it
            if cmd_str.startswith('\n'):
                cmd_str = cmd_str[1:]

            # Use shlex.split to handle quoted arguments correctly
            cmd_str_splitted = shlex.split(cmd_str)

            if len(cmd_str_splitted) < 1:
                raise ValueError(
                    "Command line format error: Missing command name")

            command_name = cmd_str_splitted.pop(0)
            command_args = dict(itertools.zip_longest(
                *[iter(cmd_str_splitted)] * 2, fillvalue=""))

            # Remove '--' from argument names
            command_args = {arg.lstrip(
                '-'): value for arg, value in command_args.items()}

            return GPTCommand(name=command_name, args=command_args)
        except Exception as e:
            # If there is any error in parsing, return an error command
            return GPTCommand(name="ERROR", args={"error": str(e)})
