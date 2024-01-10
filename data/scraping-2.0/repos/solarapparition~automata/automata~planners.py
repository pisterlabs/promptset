"""Planners for automata."""

from pathlib import Path
from typing import List, Tuple, Union

from langchain.agents import Agent

from automata.types import AutomatonAction
from automata.utilities.importing import quick_import


def default_zero_shot_planner(
    agent: Agent,
    intermediate_steps: List[Tuple[AutomatonAction, str]],
    reflection: Union[str, None],
    **kwargs,
) -> str:
    """Default planner for automata."""

    full_inputs = agent.get_full_inputs(intermediate_steps, **kwargs)
    full_inputs[
        "agent_scratchpad"
    ] = f'{full_inputs["agent_scratchpad"]}\n{reflection}\n\n{agent.llm_prefix}'
    full_output = agent.llm_chain.predict(**full_inputs)
    return full_output


def load_planner(
    automaton_path: Path, planner_info: str, request: Union[str, None] = None
) -> str:
    """Load the background knowledge for an automaton."""
    if planner_info.endswith(".py"):
        return quick_import(automaton_path / planner_info).load(request=request)
    if planner_info == "default_zero_shot_planner":
        return default_zero_shot_planner
