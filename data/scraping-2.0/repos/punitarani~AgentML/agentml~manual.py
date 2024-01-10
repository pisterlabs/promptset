"""agentml/manual.py"""

from pathlib import Path
from typing import Type
from uuid import UUID

from agentml.agents import Agent, Coder, Planner, Vision
from agentml.models import LlmMessage, LlmRole
from agentml.oai import client as openai
from agentml.sandbox import Sandbox


class Manager:
    """Manual Manager to handle different agent tasks."""

    def __init__(
        self,
        goal: str,
        csv: Path,
        session_id: UUID,
    ) -> None:
        """
        Agent constructor

        Args:
            goal (str): goal of the agent
            csv (Path): CSV file path of the dataset
            session_id (UUID): Session ID
        """

        # Ensure the CSV file exists
        if not csv.exists():
            raise FileNotFoundError(f"Agent: CSV file not found: {csv}")

        self.goal = goal
        self.csv = csv
        self.session_id = session_id

        self.sandbox = Sandbox.create(session_id=session_id, files=[csv])

        # Chat history
        self.messages: list[LlmMessage] = [
            LlmMessage(
                role=LlmRole.SYSTEM,
                content="Overarching Goal: " + self.goal,
            ),
        ]

        # Queue of tasks to run
        self.tasks: list[dict[callable, str]] = [
            {
                Planner: "Outline the steps to learn about the dataset and its features to achieve the goal"
            }
        ]

        # Stored agent instances
        self.agents = {}
        self.last_run_agent = None

    def run(self) -> list[LlmMessage]:
        """Run the agent"""

        if not self.tasks:
            print("Manager.run: No tasks to run.")
            return []

        task = self.tasks[0]

        for agent_class, objective in task.items():
            print(
                f"Manager.run: Running agent {agent_class.__name__} with objective: {objective}"
            )

            # Instantiate the agent
            agent_instance = agent_class(
                session_id=self.session_id,
                objective=objective,
                messages=[*self.messages],
            )

            # Store the agent instance
            self.agents[agent_class.__name__] = agent_instance

            # Set the last run agent
            self.last_run_agent = agent_instance

            return agent_instance.run()

    def retry_last_agent(self) -> list[LlmMessage]:
        """
        Retry the last run agent.
        """

        agent = self.last_run_agent
        if (
            agent
            and isinstance(agent, Agent)
            and (isinstance(agent, Coder) or isinstance(agent, Vision))
        ):
            print(f"Retrying agent: {type(agent).__name__}")
            return agent.retry()
        else:
            print("No suitable agent found for retry.")
            return []

    def validate_run(self, messages: list[LlmMessage]) -> None:
        """User validate the run"""

        if not self.tasks:
            print("Manager.validate_run: No tasks to validate.")
            return

        # Pop the task
        task = self.tasks.pop(0)
        print(f"Manager.validate_run: Popped task: {task}")

        # Add the messages to the chat history
        self.messages.extend(messages)

        # Handle different agents
        for agent_class, objective in task.items():
            agent_name = agent_class.__name__
            print(f"Manager.validate_run: Handling agent: {agent_name}")

            # Check if the agent instance exists
            if agent_name in self.agents:
                agent_instance = self.agents[agent_name]

                if agent_class == Planner:
                    # Add the tasks to the queue
                    print(
                        f"Manager.validate_run: Adding tasks to queue: {agent_instance.plan}"
                    )
                    self.tasks.extend(
                        [
                            {self.get_agent(task["tool"]): task["objective"]}
                            for task in agent_instance.plan
                        ]
                    )

                del self.agents[agent_name]

            else:
                print(f"Manager.validate_run: No instance found for {agent_name}")

    @staticmethod
    def next(output) -> str:
        """Get the next task"""
        next_prompt = """Based on the provided output, decide if the output is valid or invalid.
If it is invalid, return "retry",
If it is valid, return "validate",

If the output is blank, return "validate".
In most cases, you should return "validate" to continue to the next step.
Only if there is an error in the output, you should return "retry" to retry the step.
        """

        messages = [
            LlmMessage(role=LlmRole.SYSTEM, content=next_prompt),
            LlmMessage(
                role=LlmRole.USER,
                content=output,
            ),
        ]

        # Convert messages to JSON
        messages = [msg.model_dump(mode="json") for msg in messages]

        print(f"Manager.next: Sending request to OpenAI API: {messages}")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        content = response.choices[0].message.content
        print(f"Manager.next: Received response from OpenAI API: {content}")

        return content.lower()

    def done(self, output: str) -> bool:
        """Check if the manager is done"""
        done_prompt = """Based on the provided output, decide if the agent has completed the task.
Return `true` if the agent has completed the task. Otherwise, return `false`.
        """

        messages = [
            LlmMessage(role=LlmRole.SYSTEM, content=done_prompt),
            LlmMessage(
                role=LlmRole.USER, content=f"Objective to complete: {self.goal}"
            ),
            LlmMessage(
                role=LlmRole.USER,
                content=output,
            ),
        ]

        # Convert messages to JSON
        messages = [msg.model_dump(mode="json") for msg in messages]

        print(f"Manager.done: Sending request to OpenAI API: {messages}")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        content = response.choices[0].message.content
        print(f"Manager.done: Received response from OpenAI API: {content}")

        return str(content).lower() == "true"

    def add_task(self, task: dict[str, str]) -> None:
        """Add a task to the queue"""
        for agent, objective in task.items():
            print(f"Manager.add_task: Adding task: {task}")
            self.tasks.append({self.get_agent(agent): objective})

    def delete_task(self, idx: int) -> None:
        """Remove a task from the queue by index."""
        if idx < 0 or idx >= len(self.tasks):
            print(f"Invalid task index: {idx}")
            return

        removed_task = self.tasks.pop(idx)
        print(f"Removed task: {removed_task}")

    @staticmethod
    def get_agent(agent: str | Type[Agent]) -> Type[Agent]:
        """Get the agent"""

        # Check if the agent is already a class of type Agent
        if isinstance(agent, type) and issubclass(agent, Agent):
            return agent

        match agent:
            case "Coder":
                return Coder
            case "Planner":
                return Planner
            case "Vision":
                return Vision
            case _:
                raise ValueError(f"Manager.get_agent: Invalid agent: {agent}")
