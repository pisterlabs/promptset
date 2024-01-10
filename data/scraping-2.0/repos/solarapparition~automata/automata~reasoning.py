"""Type definitions for automata."""

import re
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    List,
    Sequence,
    Tuple,
    Union,
)
from typing_extensions import runtime_checkable

from langchain.agents import Agent, AgentExecutor, AgentOutputParser, ZeroShotAgent
from langchain.input import print_text
from langchain.schema import AgentFinish
from langchain.tools.base import BaseTool
from pydantic import validator

from automata.types import AutomatonAction, AutomatonStep, AutomatonReflector
from automata.validation import IOValidator


class AutomatonOutputParser(AgentOutputParser):
    """A modified version of Lanchain's MRKL parser to handle when the agent does not specify the correct action and input format."""

    final_answer_action = "Finalize Reply"
    validate_output: Union[IOValidator, None] = None

    def parse(
        self, text: str, reflection: Optional[str] = None
    ) -> Union[AutomatonAction, AgentFinish]:
        """Parse the output of the automaton."""

        # \s matches against tab/newline/whitespace
        action_regex = r"Sub-Automaton\s*\d*\s*:(.*?)\nInput\s*\d*\s*Requirements\s*\d*\s*:(.*?)\nSub-Automaton\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(action_regex, text, re.DOTALL)
        if not match:
            return AutomatonAction(
                "Think (function 0)",
                "I must determine what Sub-Automaton to delegate to, what its Input Requirements are, and what Sub-Automaton Input to send.",
                text,
                reflection,
            )
        action = match.group(1).strip()
        action_input = match.group(3)
        if self.final_answer_action in action:
            if self.validate_output is not None:
                validation_result = self.validate_output(action_input)
            return AgentFinish({"output": action_input}, text)
        return AutomatonAction(
            action, action_input.strip(" ").strip('"').strip("."), text, reflection
        )


class InvalidSubAutomaton(BaseTool):
    """Exception raised when a sub-automaton is invalid."""

    sub_automata_allowed: List[str] = []
    name = "Invalid Sub-Automaton"
    description = "Called when sub-automaton name is invalid."

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return f"{tool_input} is not a valid Sub-Automaton, try another one from the Sub-Automata list: {self.sub_automata_allowed}"

    async def _arun(self, tool_input: str) -> str:
        """Use the tool."""
        return self._run(tool_input)


@runtime_checkable
class AutomatonPlanner(Protocol):
    """Planner for automata."""

    def __call__(
        self,
        automaton_agent: Agent,
        intermediate_steps: List[Tuple[AutomatonAction, str]],
        reflection: Union[str, None],
        **kwargs,
    ) -> str:
        """Plan the next step."""


class AutomatonAgent(ZeroShotAgent):
    """Agent for automata."""

    reflect: Union[AutomatonReflector, None]
    """Reflect on information relevant to the current step."""

    planner: AutomatonPlanner
    """Plan the next step."""

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @validator("planner")
    def check_planner(cls, value):  # pylint: disable=no-self-argument
        """Check that the planner implements the `Planner` protocol."""
        if not isinstance(value, AutomatonPlanner):
            raise ValueError("`planner` must implement the `Planner` protocol.")
        return value

    def _construct_scratchpad(
        self,
        intermediate_steps: Sequence[AutomatonStep],
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""

        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\n{action.reflection}\n\n{self.llm_prefix}\n{action.log}"
            thoughts += f"\n{self.observation_prefix}{observation}"
            thoughts += "\n\n---Thoughtcycle---\n\nReflection:"
        return thoughts

    def plan(
        self, intermediate_steps: Sequence[AutomatonStep], **kwargs: Any
    ) -> Union[AutomatonAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

        reflection = (
            self.reflect(intermediate_steps, kwargs["input"]) if self.reflect else None
        )
        print_text(f"\nReflection:\n{reflection}", color="yellow", end="\n\n")
        full_output = self.planner(self, intermediate_steps, reflection, **kwargs)
        return self.output_parser.parse(full_output, reflection=reflection)


class AutomatonExecutor(AgentExecutor):
    """Executor for automata."""

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: Sequence[AutomatonStep],
    ) -> Union[AgentFinish, Sequence[AutomatonStep]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = self.agent.plan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AutomatonAction]
        if isinstance(output, AutomatonAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            self.callback_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidSubAutomaton(
                    sub_automata_allowed=list(name_to_tool_map.keys())
                ).run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color="red",
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result
