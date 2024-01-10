import re
from typing import Dict

from  langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

from virtual_assistant.prompts.input import FORMAT_INSTRUCTIONS

class TaskOutputParser(AgentOutputParser):
    @staticmethod
    def parse_all(text: str) -> Dict[str, str]:
        regex = r"Action: (.*?)[\n]Plan:(.*)[\n]What I Did:(.*)[\n]Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise Exception("parse error")

        action = match.group(1).strip()
        plan = match.group(2)
        what_i_did = match.group(3)
        action_input = match.group(4).strip(" ").strip('"')

        return {
            "action": action,
            "plan": plan,
            "what_i_did": what_i_did,
            "action_input": action_input,
        }

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Dict[str, str]:
        cleaned_output = text.strip()
        if "```yaml" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```yaml")
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json") :]
        if cleaned_output.startswith("```yaml"):
            cleaned_output = cleaned_output[len("```yaml") :]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```") :]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()

        regex = r"Action: (.*?)[\n]Plan:(.*)[\n]What I Did:(.*)[\n]Action Input: (.*)"
        match = re.search(regex, cleaned_output, re.DOTALL)
        if not match:
            raise Exception("parse error")

        parsed = TaskOutputParser.parse_all(text)
        action = parsed["action"]
        action_input = parsed["action_input"]

        if action == "Final Answer":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)

    def __str__(self):
        return "TaskOutputParser"
