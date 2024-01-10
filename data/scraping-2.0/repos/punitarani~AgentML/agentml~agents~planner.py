"""agentml/agents/planner.py"""

import json
from uuid import UUID

from agentml.models import LlmMessage, LlmRole
from agentml.oai import client as openai

from .base import Agent


class Planner(Agent):
    """Planner Agent"""

    DEFAULT_MODEL = "gpt-4-1106-preview"

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant that plans the next steps to solve a problem.
You are tasked to solve a problem using your machine learning skills: planning, reasoning, problem solving, and coding.
Based on the objective along with the optional context, you need to plan the steps and tools to solve the problem.

You have access to some tools to solve the problem.
Each tool has a specific objective which you can use to solve the problem step by step.
The objective must be a small and specific task that can be solved using the tool.

You can use the following tools:
    1. Planner - to plan and solve problems
    2. Coder - to write and execute python code
    3. Vision - to see and understand images

If a plan is not provided, explain your plan first.
Then generate the immediate next steps to solve the problem without overextending the scope.
Do not repeat tasks or steps that have already been completed successfully.
If the goal has been achieved, return an empty plan [].

Return a json object of the tool_calls array with a list of tools and objectives for each step:
    {
        "tool_calls": [
            {
                "tool": "Planner",
                "objective": "Outline the steps to load the dataset and understand the data",
            },
            {
                "tool": "Coder",
                "objective": "Load the dataset and print the first 5 rows",
            },
            {
                "tool": "Coder",
                "objective": "Print the shape of the dataset",
            },
            {
                "tool": "Vision",
                "objective": "Understand the charts and graphs to get the next steps",
            },
            {
                "tool": "Planner",
                "objective": "Outline the next steps to prepare and preprocess the data",
            },
        ]
    }

Note:
- The json object must contain only 1 key: tool_calls
- The tool_calls array must contain at least 1 item and at most 6 items
- Each item in the tool_calls array must contain 2 keys: tool and objective
- The tool key must be one of the following: Planner, Coder, Vision
- The objective key must be a string explaining the next step to solve the problem
- In most cases, the first tool will be Coder and the last tool will be Planner

Now, outline the steps to solve the problem.
    """

    def __init__(
        self,
        session_id: UUID,
        objective: str,
        messages: list[LlmMessage] = None,
        prompt: str = DEFAULT_SYSTEM_MESSAGE,
    ) -> None:
        """
        Planner Agent constructor

        Args:
            session_id (UUID): Session ID
            objective (str): Objective of the agent
            messages (list[LlmMessage], optional): List of messages to be used for the agent. Defaults to [].
            prompt (str, optional): Prompt to be used for the agent. Defaults to DEFAULT_SYSTEM_MESSAGE.
        """

        super().__init__(
            session_id=session_id, objective=objective, messages=messages, prompt=prompt
        )

        self.messages.extend(
            [
                LlmMessage(role=LlmRole.SYSTEM, content=self.prompt),
                LlmMessage(role=LlmRole.USER, content=self.objective),
            ]
        )

        # Generated plan
        self.plan: list[dict[str, str]] = []

    def run(self) -> list[LlmMessage]:
        """Run the agent"""
        print(f"Planner.run: Sending request to OpenAI API: {self.objective}")
        response = openai.chat.completions.create(
            model=self.DEFAULT_MODEL,
            messages=self.get_messages(),
            response_format={"type": "json_object"},
        )

        print(f"Planner.run: Received response from OpenAI API: {response}")
        plan = json.loads(response.choices[0].message.content).get("tool_calls", [])

        # Remove the first plan if it is Planner
        if plan and plan[0]["tool"] == "Planner":
            plan.pop(0)

        self.plan = plan

        plan_str = "\n".join(
            [f"- {task['tool']}: {task['objective']}" for task in self.plan]
        )

        messages = [
            LlmMessage(role=LlmRole.USER, content=self.objective),
            LlmMessage(
                role=LlmRole.ASSISTANT, content=f"Here is the plan:\n{plan_str}"
            ),
        ]

        return messages
