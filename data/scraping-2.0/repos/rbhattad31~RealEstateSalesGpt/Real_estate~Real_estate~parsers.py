import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish  # OutputParserException
from loguru import logger


class SalesConvoOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"  # change for salesperson_name
    verbose: bool = False

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            logger.info("In Output Parser "+text)

            if self.verbose:
                print("TEXT")
                print(text)
                print("-------")
            if f"{self.ai_prefix}:" in text:
                return AgentFinish(

                    {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
                )
            logger.info("In Output Parser 2"+text)
            print(text)
            # image_pattern = r'https?://[^\s]+'
            # image_match = re.search(image_pattern, text)
            # action = image_match.group(1)
            # action_input = image_match.group(2)
            # if image_match:
            #     return AgentAction(action.strip(), action_input.strip(" ").strip('"'),text)
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, text)
            if not match:
                ## TODO - this is not entirely reliable, sometimes results in an error.
                return AgentFinish(
                    {
                        "output": "I apologize, I was unable to find the answer to your question. Is there anything else I can help with?"
                    },
                    text,
                )
                raise OutputParserException(f"Could not parse LLM output: `{text}`")
            #logger.info("In Output Parser 3"+action)
            action = match.group(1)
            action_input = match.group(2)
            return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
        except Exception as e:
            logger.error('Ouput Parser Error: ' + str(e))


    @property
    def _type(self) -> str:
        return "sales-agent"
