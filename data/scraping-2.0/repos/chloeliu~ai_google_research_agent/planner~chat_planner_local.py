import re

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage

# from langchain_experimental.plan_and_execute.planners.base import LLMPlanner
from planner.base_local import LLMPlanner
from langchain_experimental.plan_and_execute.schema import (
    Plan,
    PlanOutputParser,
    Step,
)
### using this tools ```{tools_description}```
### You always thinking about the format in which you will compose the report so that the reader can get it with clear understanding.
### You can do detailed research on any topic and produce high quality report with comprehesive information found on the topic for the user; 
###Please make sure every step has the line: "You should include all reference data & links to back up your research;" 
# As a world class information gather, you are planning on what information you need to collect in steps to answer user's request pricisely. 
#        Each step should be about collecting information on a sub topic of the user's request.
#         - Use no more than  5 step to complete the task.
#         - Please output the plan starting with the header 'Plan:'
#         - and then followed by a numbered list of steps. 
#         - Please make the plan the minimum number of steps required to accurately complete the task. 
#         - Everystep start with "research" and end with "report".
#         - The final step should almost always be 
#         ```Report back to the user's original reques; Put the gathered information into professional report using markdown style; 
#         Use colors, style,sections, tables, headers, devisions, fonts to make the report clean and modern; References to site you used.``` 
#         - At the end of your plan, say '<END_OF_PLAN>'
        #     - The final step should almost always be 
        # ```Report back to the user's original reques; Put the gathered information into professional report using markdown style; 
        # Use colors, style, sections, tables, headers, devision to make the report clean and modern; References to site you used.``` 


SYSTEM_PROMPT = (
    """ 
    You goal is to assist the user to answer the question themselves by collecting information given the request. 
    Thinking about the top 2 more relevant topics you will gather from the internet for the user given the intentino. 
    Put the topics into a plan  in numbered steps. 
            - Please output the plan starting with the header 'Plan:'
            - Please make the plan the minimum number of steps required to accurately complete the task. 
        - At the end of your plan, say '<END_OF_PLAN>"""
)

class PlanningOutputParser(PlanOutputParser):
    """Planning output parser."""

    def parse(self, text: str) -> Plan:
        split_results = re.split("\n\s*\d+\. ", text)
        steps = [Step(value=v) for v in re.split("\n\s*\d+\. ", text)[1:]]
        return Plan(steps=steps)


def load_chat_planner(
    llm: BaseLanguageModel, tools: list, system_prompt: str = SYSTEM_PROMPT
) -> LLMPlanner:
    """
    Load a chat planner.

    Args:
        llm: Language model.
        system_prompt: System prompt.

    Returns:
        LLMPlanner
    """
    tools_description = '; '.join([f"{tool.name} - {tool.description}" for tool in tools])
    system_prompt_filled = system_prompt.format(tools_description=tools_description)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt_filled),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return LLMPlanner(
        llm_chain=llm_chain,
        output_parser=PlanningOutputParser(),
        stop=["<END_OF_PLAN>"],
    )