from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish




class MarkdownOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            
            _text = text.split("\n\n### ")[0]
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json_markdown(_text)
            action, action_input = response["action"], response["action_input"]
            if action == "final_answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, _text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, _text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": _text}, _text)

    @property
    def _type(self) -> str:
        return "conversational_chat"