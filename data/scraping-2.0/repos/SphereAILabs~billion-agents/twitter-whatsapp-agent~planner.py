from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from toolbox import ToolBox


TASK_PLANNER_SYSTEM_PROMPT = """Just do; no talk.

You are going to construct a plan to accomplish a task by breaking it down into detailed subtasks. Each step MUST use a tool provided below. List out only the steps; all other information is irrelevant.

###
You have access to these tools:
{tools}

###
use the following format:
1. [Tool]: description of the subtask
(...)"""

STEPS_PROMPT_TEMPLATE = "Steps for: '{task}'"


class TaskPlanner:
    """
    The TaskPlanner is responsible for breaking down a task into a series of subtask. It has access
    to `tools` which it can use.
    """

    def __init__(self, toolbox: ToolBox, temperature=0.7):
        self.llm = ChatOpenAI(temperature=temperature)
        self.toolbox = toolbox

    @property
    def system_prompt(self) -> str:
        system_prompt_template = PromptTemplate.from_template(
            TASK_PLANNER_SYSTEM_PROMPT
        )

        return system_prompt_template.format(tools=self.toolbox.prompt)

    def plan(self, task: str) -> str:
        steps_prompt_template = PromptTemplate.from_template(STEPS_PROMPT_TEMPLATE)
        steps_prompt = steps_prompt_template.format(task=task)

        system_message = SystemMessage(content=self.system_prompt)
        steps_message = HumanMessage(content=steps_prompt)

        messages = [system_message, steps_message]
        response = self.llm(messages)
        steps = response.content

        return steps
